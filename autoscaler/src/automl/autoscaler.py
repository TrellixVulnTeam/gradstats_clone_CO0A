import functools
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

import numpy as np
import torch
from torch.autograd import Variable
import torch.distributed as dist
from torch.optim import SGD, Optimizer
from .config import AutoScalerConfig
from apex import amp


if TYPE_CHECKING:  # pragma: no cover
    from torch.optim.optimizer import _params_t
else:
    _params_t = Any


class AdaScale(Optimizer):
    """
    Implements the AdaScale_ algorithm for scaling the learning rate for
    distributed and large batch size training. Can be used in combination with
    ``torch.nn.parallel.DistributedDataParallel`` and ``torch.optim.SGD``.

    .. _AdaScale: https://proceedings.icml.cc/static/paper_files/icml/2020/4682-Supplemental.pdf

    This class subclasses `Optimizer` so that `torch.optim.lr_scheduler` can
    work with it. In other words, AdaScale is intended to be a complete wrapper of an
    torch Optimizer.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer to apply AdaScale on
        autoscaler_cfg_path: Configuration YAML used to configure details of autoscaler
        scaler (apex or torch gradscaler): Scaler object that is being used
            for mixed precision training
        summary_writer (Tensorboard Summary Writer): Summary writer used to
            log stats for tensorboard
    """
    def __init__(self, optimizer: torch.optim.Optimizer,
                    autoscaler_cfg_path: str,
                    scaler = None,
                    summary_writer=None):
        self._optimizer = optimizer
        self._summary_writer = summary_writer 
        self._scaler = scaler

        # Proxy the param_groups so that `torch.optim.lr_scheduler` can work.
        self.param_groups = self._optimizer.param_groups
        self.cfg = AutoScalerConfig(autoscaler_cfg_path)

        self._world_size = (self.cfg.world_size if self.cfg.world_size != 0 else 
                                dist.get_world_size() if dist.is_initialized() else 1)
        self._rank = dist.get_rank() if dist.is_initialized() else 0
        # TODO: check runtime impact of updating statistics more infrequently
        self._update_interval = self.cfg.update_interval
        self._adjust_grads_for_accumulation = self.cfg.adjust_gradients_for_accumulation
        self._num_grads_to_accum = self.cfg.num_gradients_to_accumulate
        self._smoothing = self.cfg.smoothing
        if self._smoothing is None:
            self._smoothing = max(1 - self._world_size * self._num_grads_to_accum / 1000, 0)
        self._scale = self.cfg.scale
        self._scale_one_batch_size = self.cfg.scale_one_batch_size
        self._max_grad_norm = self.cfg.max_grad_norm
        self._batch_size_upper_limit = self.cfg.batch_size_upper_limit
        self._enable_debug = self.cfg.enable_debug
        self._is_adaptive = self.cfg.precondition_gradients
        self._use_pt_adam = self.cfg.use_pt_adam
        self._hook_handles: List[Any] = []
        self._hook()

        # general setup of variables internal to AdaScale functioning
        self._setup()

    def _setup(self) -> None:
        self._gain = 1.0
        self._gns = 0.0
        self._effective_lr = 0.0
        self._real_iterations = 0
        self._local_grad_sqr: Optional[torch.Tensor] = None
        self._num_backward_calls = 0
        self._last_final_backward_call = 0        
        self._gain_invalid = True
        self._num_param_groups = len(self._optimizer.param_groups)
        # Populate state dictionary with AdaScale stats
        # We are going to track the following
        # 1. per-param-group sqr & var states
        # 2. scale invariant steps - so that we can track how much progress we
        #    made even when the scale of training has changed
        # 3. What else? - depends on experiments 
        self._optimizer.state.setdefault(
            "adascale",
            {
                "scale_invariant_steps": 0.0,
                "grad_sqr_avg": np.ones(self._num_param_groups),
                "grad_var_avg": np.zeros(self._num_param_groups),
            },
        )

        # TODO: is this required? - maybe can access these hps directly from self._optimizer
        if self._is_adaptive:
            self._opt_param_group = {'beta1': [], 'beta2': [], 'eps': []}
            for pg_idx, param_group in enumerate(self._optimizer.param_groups):
                self._opt_param_group['beta1'].append(param_group['betas'][0])
                self._opt_param_group['beta2'].append(param_group['betas'][1])
                self._opt_param_group['eps'].append(param_group['eps'])

        # Adding for O2 level of AMP
        self.state = self._optimizer.state
        self.local_grad_sqr = None
        # this tracks the sum of adascale steps taken so far and is used to estimate
        # speed-ups obtained by scaling
        self._scale_invariant_steps = 0.0
        # stability related constants for ADAM with AdaScale
        self._SAFE_UPDATE_RATIO = 1.0
        self._MIN_STEPS = 50

    def _hook(self) -> None:
        """ Internal function to register the gradient hooks.

            Note, don't assume every parameter will generate a gradient (i.e. triggering the hook)
            in every backward pass, which is the reason that we have ``find_unused_params`` flag
            in the DDP class in ``torch.nn.parallel``.
        """
        assert self._hook_handles == [], "Must run unhook first"
        for pg_idx, param_group in enumerate(self._optimizer.param_groups):
            for param in param_group["params"]:
                h = param.register_hook(functools.partial(self._backward_hook, pg_idx, param))
                self._hook_handles.append(h)

    def __del__(self) -> None:
        """ Unhook in case caller forgets to call unhook.
            This however may not "work" since there would be circular reference
            between the hook objects and this objects. In that case, neither will
            get GC'ed. Calling unhook explicitly if you really want to delete
            AdaScale from memory.
        """
        self.unhook()

    def unhook(self) -> None:
        """ Unregister hook handles.
            This is public because caller may need to call this to ensure all GPU
            memory are released. Otherwise, the hook may prevent parameters from being
            released from the GPU memory pool.

            Internally, we use this to support ``add_param_group()`` API.
        """
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

    @property
    def _state(self) -> Dict[str, np.ndarray]:
        """
        Return the state of AdaScale.
        """
        return self._optimizer.state["adascale"]

    @property
    def scale(self) -> float:
        """
        The scaling factor of the current batch size, relative to the baseline
        batch size, which could be a DDP training. For example, if the
        baseline batch size is 32 on 2 GPUs, but using a scaled-up batch size
        of 80 on 4 GPUs, then then the scaling factor is 80 * 4 / 32 / 2 = 5.

        This is exposed API mainly for logging purpose. Note, this is different
        from ``self.gain()``.

        Returns:
            (float):
                The current scaling factor.
        """
        return self._scale

    @property
    def smoothing(self) -> float:
        """
        The smoothing constant used in exponentially-weighted moving average
        tracking the gradient norm mean and variance within AdaScale.

        This is exposed API since the value is computed and caller may
        want to obtain this value and log it.

        Returns:
            (float):
                The current smoothing value.
        """
        return self._smoothing

    def set_scale(self, scale: float, update_estimate: bool = True) -> None:
        """
        Set the scaling factor of the current batch size. It is up to the
        application to invoke this function to make sure that AdaScale's
        scaling factor matches the actual batch size used during training.

        Args:
            scale (float):
                New scaling factor to be applied to AdaScale.
            update_estimate (bool):
                Whether to update the scale-depenent estimate of gradient
                variance; this is highly recommended. (default: True)
        """
        assert self._local_grad_sqr is None, "Don't change scale in backward phase"
        assert scale >= 1, "Scale must be at least 1"
        if update_estimate and hasattr(self, "_scale"):
            assert self._scale >= 1, "bug: old scale isn't valid"
            # Rescale grad_var_avg to account for the change in scale
            if "grad_var_avg_biased" in self._state:
                self._state["grad_var_avg_biased"] *= self._scale / scale
            self._state["grad_var_avg"] *= self._scale / scale
        self._scale = scale

    def _grad_sqr_avg(self, pg_idx: Optional[int] = None) -> float:
        """
        Current estimate of the squared l2-norm of the true gradient
        (mu squared in the AdaScale paper).

        Args:
            pg_idx (Optional[int]):
                Optional index for a parameter group.

        Returns:
            (float):
                Estimate of squared l2-norm.
        """
        if pg_idx is not None:
            return self._state["grad_sqr_avg"][pg_idx]
        else:
            return float(np.sum(self._state["grad_sqr_avg"]))

    def _grad_var_avg(self, pg_idx: Optional[int] = None) -> float:
        """
        Current estimate of the trace of the covariance of the true gradient
        (sigma squared in the AdaScale paper).

        Args:
            pg_idx (Optional[int]):
                Optional index for a parameter group.

        Returns:
            (float):
                Estimate of trace of the covariance.
        """
        if pg_idx is not None:
            return self._state["grad_var_avg"][pg_idx]
        else:
            return float(np.sum(self._state["grad_var_avg"]))

    def scale_invariant_steps(self, pg_idx: Optional[int] = None) -> float:
        """
        This is the number of steps we advance scheduler by per optimizer step.
        For aggressive schedules like cosine decay we use a heuristic to make the
        adapted schedule aggressive as well

        Args:
            pg_idx (int):
                Optional index of a parameter group.
                Default None: returns "averaged" gain for all groups.

        Returns:
            (float):
                Estimate of gain ratio.
        """
        if self._gain_invalid:
            return 1.0
        var = self._grad_var_avg(pg_idx)
        sqr = self._grad_sqr_avg(pg_idx)
        gain = (var + sqr) / (var / self.scale + sqr)
        if self.cfg.aggressive_schedule:
            # take larger scheduler steps to maintain the aggressive schedule
            return np.power(self.scale * self.scale * gain, 1./3)
        return gain

    def gain(self, pg_idx: Optional[int] = None, alpha=0.5) -> float:
        """
        Current estimate of the AdaScale gain ratio (r_t in the paper).

        Args:
            pg_idx (int):
                Optional index of a parameter group.
                Default None: returns "averaged" gain for all groups.

        Returns:
            (float):
                Estimate of gain ratio.
        """
        var = self._grad_var_avg(pg_idx)
        sqr = self._grad_sqr_avg(pg_idx)
        # for tensorboard
        self._var = var
        self._sqr = sqr
        if self._gain_invalid:
            # in case there is no gain - we backoff to base case
            self._gain = 1.0
            return 1.0
        max_scale = self.scale
        if self._is_adaptive:
            max_scale = max_scale**alpha
        gain = (var + sqr) / (var / max_scale + sqr)
        self._gain = gain
        return gain

    def gns(self, pg_idx: Optional[int] = None) -> float:
        """
        Computes GNS as B_simple defined in https://arxiv.org/pdf/1812.06162.pdf

        AdaScale calculations already take into account computing trace(cov)/batch_size estimate and squared
        of gradient norm.

        We can estimate b_simple = grad_var * batch_size / grad_sqr
        NOTE: that batch size used here is batch size that corresponds to scale 1.0
        """
        print(self._real_iterations, self._scale_invariant_steps)
        # TODO: compare numbers with original estimator in the paper
        if self._real_iterations < self._MIN_STEPS:
            # allow averages to stabilize before predicting
            return self._scale_one_batch_size
        if self._gain_invalid:
            # fall back to base scale case
            self._gns = self._scale_one_batch_size
            return self._scale_one_batch_size
        var = self._grad_var_avg(pg_idx)
        sqr = self._grad_sqr_avg(pg_idx)
        if self._enable_debug:
            print("IN GNS (sqr, var):", sqr, var)
        gns = self._scale_one_batch_size * var / sqr
        # clip GNS for upper limit
        gns = min(gns, self._batch_size_upper_limit)
        self._gns = gns
        return gns

    def _update_avg(self, name: str, value: np.ndarray, factor: float) -> None:
            # This function computes and stores the moving average of a vector
            # using a smoothing factor.
            biased = self._state.get(name + "_biased", np.zeros(value.shape[0]))
            unbias = self._state.get(name + "_unbias", np.zeros(value.shape[0]))
            biased = factor * biased + (1.0 - factor) * value
            unbias = factor * unbias + (1.0 - factor)
            self._state[name + "_biased"] = biased
            self._state[name + "_unbias"] = unbias
            self._state[name] = biased / unbias

    def _current_loss_scale(self):
        return self._scaler.get_scale() if self._scaler else amp.state_dict()['loss_scaler0']['loss_scale']

    def _get_norm_squared(self, pg_idx, param, grad):
        # unscale grads before computing squares - else numbers blow up with scale
        curr_loss_scale_squared = self._current_loss_scale()**2
        preconditioner = self._calculate_preconditioner(pg_idx, param)
        divisor = preconditioner * curr_loss_scale_squared
        norm = torch.nan_to_num(torch.linalg.norm(grad.div(divisor)))
        return norm ** 2

    def _total_grad_sqr(self):
        curr_loss_scale = self._current_loss_scale()
        # colocate total sqr with local sqr tensor
        total_grad_sqr = torch.zeros_like(self._local_grad_sqr)

        for pg_idx, param_group in enumerate(self._optimizer.param_groups):
            for param in param_group["params"]:
                # we are going to exclude missing or NaN values in gradients - note avoiding setting NaN to 0.0
                if param.grad is None or torch.any(torch.isnan(param.grad)):
                    continue
                total_grad_sqr[pg_idx] += self._get_norm_squared(pg_idx, param, param.grad)
        # EXPERIMENTAL CLAMP squared values to avoid blow-up - note we do not modify grads
        # but just the piece that computes stats
        total_grad_sqr = torch.clamp(total_grad_sqr, min=0.0, max=1e11)
        return total_grad_sqr


    def _backward_hook(self, pg_idx: int, param: torch.Tensor, grad: torch.Tensor) -> None:
        # This method should be invoked once for each parameter during the
        # backward pass, before gradients are synchronized between world_size.
        grads_are_invalid = False
        if torch.sum(torch.isnan(grad)) or torch.sum(torch.isinf(grad)):
            grads_are_invalid = True

        # Store the local gradient square sums in a tensor colocated with grad
        # This vector is also used for error checking. Whenever it is not None,
        # it means that we are in backward pass.
        if self._local_grad_sqr is None:
            self._local_grad_sqr = torch.zeros(len(self._optimizer.param_groups),
                                                device=grad.device,
                                                requires_grad=False,
                                                dtype=torch.float64)

        # we want accum copies of local_grad_sqr per worker 
        if not grads_are_invalid:
            self._local_grad_sqr[pg_idx] += self._get_norm_squared(pg_idx, param, grad) 
        # Now, ensure we queue a callback at the end of the callback queue.
        # This will fire after all gradient callbacks are done (esp. those
        # queued by DDP.
        self._final_callback_queued = False
        Variable._execution_engine.queue_callback(self._queue_callback)

    def _queue_callback(self) -> None:
        # This method should be invoked after the entire backward pass. We want
        # to make sure self._final_callback is invoked once, only after all
        # gradients have been synchronized between each worker. However, the
        # synchronization code in DistributedDataParallel is also done in a
        # callback, which might not yet be executed. Therefore, we enqueue
        # self._final_callback from this method, which should ensure it is
        # invoked after the gradient synchronization callback.
        if self._final_callback_queued:
            return
        self._final_callback_queued = True
        Variable._execution_engine.queue_callback(self._final_callback)

    def _final_callback(self) -> None:
        # This method should be invoked once for each backward pass, after
        # gradients have been synchronized between each worker, unless we
        # are in gradient accumulation mode, where grads are not all_reduced
        # between the GPUs.
        self._final_callback_queued = False
        assert isinstance(self._local_grad_sqr, torch.Tensor)
        # Keep track of number of backward calls for gradient accumulation.
        self._num_backward_calls += 1
        assert (self._num_backward_calls - self._last_final_backward_call) <= self._num_grads_to_accum,\
            (f"bug: {self._num_backward_calls} - {self._last_final_backward_call} should <= {self._num_grads_to_accum}")
        if (self._num_backward_calls - self._last_final_backward_call) % self._num_grads_to_accum != 0:
            assert self._local_grad_sqr is not None, "We should still be in backward phase"
            return

        # This vector has length of # of param_groups
        work = None
        # EXPERIMENTAL CLAMP squared values to avoid blow-up - note we do not modify grads
        # but just the piece that computes stats
        self._local_grad_sqr = torch.clamp(self._local_grad_sqr, min=0.0, max=1e11)
        # we store the squared norm at local level before allreduce
        np_local_grad_sqr = self._local_grad_sqr.clone().cpu().numpy()
 
        # check for large outliers - don't apply to moving averages if "very" large
        found_outlier = False
        if self.local_grad_sqr is None:
            self.local_grad_sqr = np_local_grad_sqr

        if self._enable_debug:
            print("rank={}, latest={}, previous={}".format(self._rank, np_local_grad_sqr, self.local_grad_sqr))

        if  self._real_iterations > self._MIN_STEPS and self.local_grad_sqr[0] > 0.0 and \
                np.abs(np.log10(np_local_grad_sqr[0]/self.local_grad_sqr[0])) > self._SAFE_UPDATE_RATIO:
            found_outlier = True
            # use previous value
            for i, v in enumerate(self.local_grad_sqr):
                self._local_grad_sqr[i] = self.local_grad_sqr[i]
            if self._enable_debug:
                print("OUTLIER detected ratio={}, reusing previous value".format(np.abs(np.log(np_local_grad_sqr[0]/self.local_grad_sqr[0]))))
        else:
            self.local_grad_sqr = np_local_grad_sqr

        if self._world_size > 1:
            work = dist.all_reduce(self._local_grad_sqr, async_op=True)

        total_grad_sqr = self._total_grad_sqr()
        # Divide by (_num_grads_to_accum ** 2) to account for gradient
        # accumulation. Note that sometimes this factor is already taken care of in
        # loss calculation, so we do not need to adjust for accumulation divisor
        if self._num_grads_to_accum > 1 and self._adjust_grads_for_accumulation:
            total_grad_sqr = total_grad_sqr / (self._num_grads_to_accum ** 2)

        total_grad_sqr = total_grad_sqr.cpu().numpy()
        # Wait for all_reduce to be done and move it to cpu & np.
        if work:
            work.wait()
        local_grad_sqr = self._local_grad_sqr.cpu().numpy()

        # save as object variable only for Tensorboard logging
        self.total_grad_sqr = total_grad_sqr

        # See appendix B.3 of the paper.
        # Modified to handle cases where scale != world_size
        #
        # local_grad_sqr is \sum_{i=1}^{c N} \norm{g_t_i}^2
        # where N is world size and c is num_grads_to_accum
        # total_grad_sqr is \norm{\bar{g}_t}^2
 
        # adjusting stats for original formula
        # the reasoning for this adjustment is as follows: if we adjusted the accumulation factor as 
        # a predivision (in loss calc - as in our BERT codebase) then we are scaling each "local" grad
        # vector by the accum factor, which we do not want - the accum factor should only affect the
        # all reduced (large batch gradient.)
        if not self._adjust_grads_for_accumulation:
            local_grad_sqr = local_grad_sqr * (self._num_grads_to_accum**2)

        S = self._scale

        # TODO: write tests to check this (verify accumulation mechanism interaction with DDP hook)
        # When accumulating gradients we have `accum_steps` number of gradients per worker
        # which is further summed to get `local_grad_sqr` - total of `accum_steps`*`num_workers`
        # gradients sample size is `accum_steps`*`num_workers`
        # `total_grad_sqr` is squared l2-norm of allreduced gradient

        cN = self._world_size * self._num_grads_to_accum
        # when S = cN the formulation reduces to that in paper
        if S > 1:
            grad_var = local_grad_sqr * (S / cN) / (cN - 1) - total_grad_sqr * S / (cN - 1)
            grad_sqr = total_grad_sqr - grad_var / S
        else:
            grad_var = local_grad_sqr / (cN - 1) - total_grad_sqr * cN / (cN - 1)
            grad_sqr = total_grad_sqr - grad_var / cN

        # Bounding these values artificially is not good - affects moving averages which in turn lingers on depending on smoothing
        #grad_var = np.maximum(grad_var, 1e-6)
        #grad_sqr = np.maximum(grad_sqr, 0.0)
        if self._enable_debug and self._rank == 0:
            print('sqr:', grad_sqr, 'var:', grad_var)

        # for tensorboard (mostly to catch abnormal values, for all calculations smoothed values are used)
        self._nonsmooth_var = grad_var
        self._nonsmooth_sqr = grad_sqr

        self._gain_invalid = False
        if np.any( grad_var <= 0.) or np.any( grad_sqr <= 0.) or np.isnan(np.sum(grad_sqr)) or np.isinf(np.sum(grad_sqr)):
            if self._enable_debug and self._rank == 0:
                print('gradient inf/nan skipping update of moving averages of grad moments', grad_var, grad_sqr)
                print(local_grad_sqr, S, cN, total_grad_sqr, self._current_loss_scale())
            self._gain_invalid = True
        else:
            self._update_avg("grad_sqr_avg", grad_sqr, self.smoothing)
            self._update_avg("grad_var_avg", grad_var, self.smoothing)

        # reset backward call counters for next param update cycle
        self._last_final_backward_call = self._num_backward_calls = 0
        # Indicating backward is done.
        self._local_grad_sqr = None


    def get_step_increment(self):
        """
        Step increment is an integer that is used by the scheduler to move forward in
        the training loop
        """
        assert self._local_grad_sqr is None, "Don't step without finishing backward phase"
        if self._gain_invalid:
            return 1 # should this be 1 or 0
        prev_steps = np.floor(self._scale_invariant_steps)
        self._scale_invariant_steps += self.scale_invariant_steps()
        step_increment = np.floor(self._scale_invariant_steps) - prev_steps
        self._real_iterations += 1
        return int(step_increment)


    def step(self, *args: Any, **kwargs: Any) -> Optional[float]:
        """
        Run one optimizer step using Adascale. Essentially just invokes
        ``optimizer.step(*args, **kwargs)`` with a scaled learning rate.

        .. note::

            It is possible that this function becames a performance
            bottleneck if you have frequent updates. To avoid that,
            making bigger steps and reducing update frequency is generally
            better for performance.

        Args:
            args (Any):
                Positional arguments passed to ``optimizer.step``.
            kwargs (Any):
                Keyword arguments passed to ``optimizer.step``.

        Returns:
            (Tensor):
                The loss tensor if a closure if used to re-evaluate the model.
        """
        assert self._local_grad_sqr is None, "Don't step without finishing backward phase"
        # Set original LR and set new LR.
        original_lr = []
        for pg_idx, param_group in enumerate(self._optimizer.param_groups):
            original_lr.append(param_group["lr"])
            param_group["lr"] = self.gain(pg_idx=pg_idx) * param_group["lr"]
            # log effective LR for param group 0
            if pg_idx == 0:
                self._effective_lr = param_group["lr"]
        res = None
        # Step it.
        if self._scaler:
            res = self._scaler.step(self._optimizer)
        else:
            self._optimizer.step(*args, **kwargs)
        # Restore the original LR.
        for lr, param_group in zip(original_lr, self._optimizer.param_groups):
            param_group["lr"] = lr
        # FIXME: AMP O2 seems to create a copy of param group dicts, so the proxy setup in c-tor breaks, force resync here so that scheduler works properly
        self.param_groups = self._optimizer.param_groups
        return res

    def add_param_group(self, pg: Dict) -> None:
        """ Support adding parameter groups

            We need to re-size some of the state and re-register the backward hooks.
        """
        assert self._local_grad_sqr is None, "Can't add parameter group during backward"
        self._optimizer.add_param_group(pg)
        # Update the hooks.
        self.unhook()
        self._hook()
        # Extend the states.
        for name in self._state.keys():
            assert name.startswith("grad_sqr_avg") or name.startswith("grad_var_avg"), name
            if name.endswith("_count"):
                # This is the "_count" variable, should be a 1D int.
                assert self._state[name].shape == (1,), self._state[name].shape
                continue
            # must be a np array, extend it with the right value and check the shape.
            val = 1 if name == "grad_sqr_avg" else 0
            self._state[name] = np.append(self._state[name], val)
            assert self._state[name].shape == (len(self._optimizer.param_groups),)

    def zero_grad(self) -> None:
        """Proxy function to optimizer, because some training loops need this."""
        assert self._local_grad_sqr is None, "Don't zero_grad in backward"
        return self._optimizer.zero_grad()

    def state_dict(self) -> Dict:
        """ Proxy function to optimizer, checkpointing needs this.

            .. note::

                Do NOT checkpoint in the middle of gradient accumulation since
                associated AdaScale internal states are not saved in the checkpoint.
        """
        assert self._local_grad_sqr is None, "Don't checkpoint in backward"
        return self._optimizer.state_dict()


    def load_state_dict(self, data: Dict) -> None:
        """ Proxy function to optimizer, checkpointing needs this.

            .. note::

                Do NOT checkpoint in the middle of gradient accumulation since
                associated AdaScale internal states are not saved in the checkpoint.
        """
        assert self._local_grad_sqr is None, "Don't load checkpoint in backward"
        return self._optimizer.load_state_dict(data)

    def rescale_training_cluster(self, scale:int, num_gradients_to_accumulate: int, smoothing: float = True,) -> None:
        """
        TODO: This is probably the entrypoint for training manager to indicate update in scale
        """
        raise NotImplementedError

    def _calculate_preconditioner(self, pg_idx, param, where="local"):
        """
        From openai paper - One might also use preconditioned gradients, obtained for example by dividing gradient 
        components by the squareroot of the Adam optimizer’s [KB14] accumulated variances.
        in case of ADAM - note that averages won't be very useful until we have done 1/(1-beta2) batches, so we
        ignore batch size predictions initially
        Q. should we not precondition for the initial steps? How does this affect AdaScale stats??
        TODO: Investigate other preconditioners
        """
        if self._real_iterations < self._MIN_STEPS or \
                not self._is_adaptive or \
                param not in self._optimizer.state:
            return torch.ones_like(param, memory_format=torch.preserve_format)
        # get current state for param
        state = self._optimizer.state[param]
        # get param group settings
        group = self._optimizer.param_groups[pg_idx]
        _, beta2 = group['betas']
        # check of using Apex FusedADAM or PT Adam/W
        step = group['step'] if not self._use_pt_adam else state['step'] 
        exp_avg_sq = state["exp_avg_sq"]
        eps = self._opt_param_group['eps'][pg_idx]
        bias_correction = 1 - beta2 ** step
        pinv = (exp_avg_sq / bias_correction).sqrt().add_(eps)
        return pinv

    def log_to_tensorboard(self, real_iteration):
        self._summary_writer.add_scalar('Train/Real Iterations', real_iteration, self._scale_invariant_steps)
        self._summary_writer.add_scalar('Train/Gain', self._gain, self._scale_invariant_steps)
        self._summary_writer.add_scalar('Train/var', self._nonsmooth_var[0], self._scale_invariant_steps)
        self._summary_writer.add_scalar('Train/sqr', self._nonsmooth_sqr[0], self._scale_invariant_steps)
        self._summary_writer.add_scalar('Train/var_smooth', self._var, self._scale_invariant_steps)
        self._summary_writer.add_scalar('Train/sqr_smooth', self._sqr, self._scale_invariant_steps)
        self._summary_writer.add_scalar('Train/allreduced_grad_sqr', self.total_grad_sqr[0],self._scale_invariant_steps)
        self._summary_writer.add_scalar('Train/local_grad_sqr', self.local_grad_sqr[0],self._scale_invariant_steps)
        self._summary_writer.add_scalar('Train/GNS', self._gns, self._scale_invariant_steps)
        self._summary_writer.add_scalar('Train/Effective LR', self._effective_lr, self._scale_invariant_steps)
        self._summary_writer.flush()

