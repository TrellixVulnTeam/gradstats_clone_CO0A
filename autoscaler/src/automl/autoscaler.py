import functools
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

import numpy as np
import torch
from torch.autograd import Variable
import torch.distributed as dist
from torch.optim import SGD, Optimizer

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
        optimizer (torch.optim.Optimizer):
            Optimizer to apply AdaScale to.
        world_size (int):
            Number of world_size for distributed training.
            If None, defaults to ``dist.get_world_size()``.
        scale (float):
            Scaling factor of the batch size from scale equals 1, e.g. using a 10x
            larger batch size (summed across all ranks with gradient accumulation)
            means a scale of 10.
            If None, defaults to ``world_size * num_gradients_to_accumulate``.
        smoothing (float):
            Smoothing factor for moving average.
            If None, it defaults to ``max(1 - (world_size * num_gradients_to_accumulate)/1000, 0)``.
        num_gradients_to_accumulate (int):
            Number of passes that we accumulate gradients locally
            between each optimizer step. This can be changed during
            training as long as the train loop changes gradient accumulation
            accordingly.
            Default to 1, which does not accumulate gradients.
        debias_ewma (bool):
            (experimental) Use debias exponential moving average
            for smoothing and mu and sigma variables. False will
            use the method in the paper's Appendix B.3.
            Default: True, which is what have been validated so far.
        rank (int):
            Rank of the worker (for debugging purposes only)
        is_adaptive (bool):
            True if using adaptive first order optimizer, currently supports NovoGrad
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        world_size: Optional[int] = None,
        scale: Optional[float] = None,
        smoothing: float = None,
        num_gradients_to_accumulate: int = 1,
        debias_ewma: bool = True,
        rank: int = 1,
        is_adaptive:bool = False,
        scaler = None,
        adjust_grads_for_accumulation = False,
        use_preconditioner = False,
        summary_writer=None,
        model=None # for gradient clipping in case we detect overflows
    ):
        self._optimizer = optimizer
        self._local_grad_sqr: Optional[torch.Tensor] = None
        self._world_size: int = (
            world_size if world_size is not None else dist.get_world_size() if dist.is_initialized() else 1
        )
        self._num_backward_calls = 0
        self._last_final_backward_call = 0
        self._num_grads_to_accum = num_gradients_to_accumulate
        self._debias_ewma = debias_ewma
        self._rank = rank
        self._is_adaptive = is_adaptive
        # NOTE: If using nccl then this has to be a cuda tensor
        self._gain_invalid = torch.ones(1, dtype=torch.uint8, requires_grad=False).cuda() #True
        # Proxy the param_groups so that `torch.optim.lr_scheduler` can work.
        self.param_groups = self._optimizer.param_groups
        self._smoothing = smoothing
        self.set_num_gradients_to_accumulate(num_gradients_to_accumulate, update_smoothing=smoothing is None)
        self._adjust_grads_for_accumulation = adjust_grads_for_accumulation
        self._use_preconditioner = use_preconditioner
        self.summary_writer = summary_writer
        self._model = model

        if self._world_size * self._num_grads_to_accum <= 1:
            # gain will be NaN since we will be dividing by zero in paper's B.3 where (S-1) == 0.
            raise RuntimeError("AdaScale does not support a single worker without grad accumulation.")

        # Per-param-group sqr & var states (sigma^2 & mu^2 in the paper).
        self._optimizer.state.setdefault(
            "adascale",
            {
                "grad_sqr_avg": np.ones(len(optimizer.param_groups)),
                "grad_var_avg": np.zeros(len(optimizer.param_groups)),
            },
        )

        self._scale = 1.0  # Assign to inform mypy about the typing of this variable.
        self.set_scale(self._world_size * self._num_grads_to_accum if scale is None else scale)

        # FIXME: write more generic - MAYBE THIS IS NOT NEEDED
        if self._is_adaptive:
            self._opt_param_group = {'beta1': [], 'beta2': [], 'eps': []}
            for pg_idx, param_group in enumerate(self._optimizer.param_groups):
                self._opt_param_group['beta1'].append(param_group['betas'][0])
                self._opt_param_group['beta2'].append(param_group['betas'][1])
                self._opt_param_group['eps'].append(param_group['eps'])

        self._hook_handles: List[Any] = []
        self._hook()
        self._scaler = scaler
        # Adding for O2 level of AMP
        self.state = self._optimizer.state
        self.local_grad_sqr = None
        # WIP - steps (should be part of state) - maybe we should track steps inside this class - CHECK
        self.steps = 0

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
        Return the states of AdaScale.
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
            if self._debias_ewma and "grad_var_avg_biased" in self._state:
                self._state["grad_var_avg_biased"] *= self._scale / scale
            elif "grad_var_avg_total" in self._state:  # _debias_ewma==False
                self._state["grad_var_avg_total"] *= self._scale / scale
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

    def scale_invariant_steps(self, pg_idx: Optional[int] = None, aggressive_base_schedule=False) -> float:
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
        if self._gain_invalid[0] != 0:
            return 0.0 # don't update model weights
        var = self._grad_var_avg(pg_idx)
        sqr = self._grad_sqr_avg(pg_idx)
        gain = (var + sqr) / (var / self.scale + sqr)
        if aggressive_base_schedule:
            #return np.sqrt(self.scale * gain)
            # take larger scheduler steps to maintain the aggressive schedule
            return np.power(self.scale * self.scale * gain, 1./3)
        return gain

    def gain(self, pg_idx: Optional[int] = None, power_law_ratio=0.5) -> float: #power_law_ratio=0.618) -> float:
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
        self.var = var
        self.sqr = sqr
        if self._gain_invalid[0] != 0:
            return 0.0
        max_scale = self.scale
        if self._is_adaptive:
            max_scale = np.power(max_scale, power_law_ratio)
        gain = (var + sqr) / (var / max_scale + sqr)
        return gain


    def gns(self, scale_one_batch_size=32, pg_idx: Optional[int] = None) -> float:
        """
        Computes GNS as B_simple defined in https://arxiv.org/pdf/1812.06162.pdf

        AdaScale calculations already take into account computing trace(cov)/batch_size estimate and squared
        of gradient norm.

        We can estimate b_simple = grad_var * batch_size / grad_sqr
        NOTE: that batch size used here is batch size that corresponds to scale 1.0
        """
        # TODO: compare numbers with original estimator in the paper
        if self._gain_invalid[0] != 0:
            return 0.0 # AS: return some value that makes gns unusable for this iteration
        # estimate of grad var for scale S
        var = self._grad_var_avg(pg_idx)
        sqr = self._grad_sqr_avg(pg_idx)
        gns = scale_one_batch_size * var / sqr
        # TODO: clip GNS for upper limit
        return gns


    def _update_avg(self, name: str, value: np.ndarray, factor: float) -> None:
        if self._debias_ewma:
            # This function computes and stores the moving average of a vector
            # using a smoothing factor.
            biased = self._state.get(name + "_biased", np.zeros(value.shape[0]))
            unbias = self._state.get(name + "_unbias", np.zeros(value.shape[0]))
            biased = factor * biased + (1.0 - factor) * value
            unbias = factor * unbias + (1.0 - factor)
            self._state[name + "_biased"] = biased
            self._state[name + "_unbias"] = unbias
            self._state[name] = biased / unbias
        else:
            # Moving average procedure described in Appendix B.3
            # For iterations t < 1 / (1 - smoothing) define grad_var_avg
            # and grad_sqr_avg as mean of the past samples. After that
            # start using running average.
            #
            # Note: we only keep a single _count for all parameter groups.
            #       Ideally, it should be a vector and in case a PG is added
            #       after some iterations are done. But, then the if condition
            #       below will need to be a np.where. I leave this corner
            #       case to a future exercise.
            count = self._state.get(name + "_count", np.zeros(1))
            count[0] += 1
            self._state[name + "_count"] = count
            if count < 1 / (1 - self._smoothing):
                total = self._state.get(name + "_total", None)
                if total is None:
                    total = value
                else:
                    total += value
                self._state[name + "_total"] = total
                self._state[name] = total / count
            else:
                self._state[name] = factor * self._state[name] + (1.0 - factor) * value

    def _current_loss_scale(self):
        return self._scaler.get_scale() if self._scaler else amp.state_dict()['loss_scaler0']['loss_scale']


    def _get_norm_squared(self, pg_idx, param, grad):
        # unscale grads before computing squares - else numbers blow up with scale
        curr_loss_scale_squared = self._current_loss_scale()**2
        preconditioner = self._calculate_preconditioner(pg_idx, param)
        divisor = preconditioner * curr_loss_scale_squared
        norm = torch.nan_to_num(torch.linalg.norm(grad.div(divisor)))
        return norm * norm


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
        # total_grad_sqr = torch.clamp(total_grad_sqr, min=0.0, max=1e11)
        
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
        # self._local_grad_sqr = torch.clamp(self._local_grad_sqr, min=0.0, max=1e11)
        
        # we store the squared norm at local level before allreduce
        np_local_grad_sqr = self._local_grad_sqr.clone().cpu().numpy()
        
        # check for large outliers - don't apply to moving averages if "very" large
        found_outlier = False
        SAFE_RATIO = 10.0
        MIN_STEPS = 50
        if self.local_grad_sqr is None:
            self.local_grad_sqr = np_local_grad_sqr
        print("rank={}, latest={}, previous={}".format(self._rank, np_local_grad_sqr, self.local_grad_sqr))
        if  self.steps > MIN_STEPS and self.local_grad_sqr[0] > 0.0 and np_local_grad_sqr[0]/self.local_grad_sqr[0] > SAFE_RATIO:
            found_outlier = True
            print("OUTLIER detected ratio={}, skipping optimizer state update".format(np_local_grad_sqr[0]/self.local_grad_sqr[0]))
#            # use previous value
#            for i, v in enumerate(self.local_grad_sqr):
#                self._local_grad_sqr[i] = self.local_grad_sqr[i]
#        else:
#            self.local_grad_sqr = np_local_grad_sqr
        self.local_grad_sqr = np_local_grad_sqr



        if self._world_size > 1:
            work = dist.all_reduce(self._local_grad_sqr, async_op=True)  # SUM

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
        if not self._adjust_grads_for_accumulation:
            local_grad_sqr = self._num_grads_to_accum * self._num_grads_to_accum * local_grad_sqr

        S = self._scale
        # AS: accum taken care of during loss calc - here we have `num workers` copies of local_sqr and 
        # but total_sqr is square of average of `accum steps` * `num workers` batches
        cN = self._world_size * self._num_grads_to_accum
        # cN = self._world_size
        # AS: Adjustment is done as such
        # S/(cN-1) * (1/cN * \sum_{i=1}^cN \norm{g_t_i}^2 - \norm{\bar{g}_t}^2)
        grad_var = local_grad_sqr * (S / cN) / (cN - 1) - total_grad_sqr * S / (cN - 1)
        # grad_var = local_grad_sqr * (S / cN) / (cN - 1) - total_grad_sqr / (self._world_size * self._num_grads_to_accum - 1)
        grad_sqr = total_grad_sqr - grad_var / S

        # for tensorboard (mostly to catch abnormal values, for all calculations smoothed values are used)
        self.nonsmooth_var = grad_var
        self.nonsmooth_sqr = grad_sqr

        self._gain_invalid[0] = 0
        if found_outlier or np.any( grad_var <= 0.) or np.any( grad_sqr <= 0.) or np.isnan(np.sum(grad_sqr)) or np.isinf(np.sum(grad_sqr)):
            print('local: gradient inf/nan skipping update of moving averages of grad moments')
            self._gain_invalid[0] = 1

        # an extra boolean sync here for checking if any of the worker stats blew up and skip update
        if self._world_size > 1:
            dist.all_reduce(self._gain_invalid)

        # if grads are valid on all workers then update moving averages
        if self._gain_invalid[0] == 0:
            self._update_avg("grad_sqr_avg", grad_sqr, self.smoothing)
            self._update_avg("grad_var_avg", grad_var, self.smoothing)
        else:
            print('global: gradient inf/nan skipping update of moving averages of grad moments')

        if self._rank == 0:
            print("rank:", self._rank, "grad_var:", grad_var, "grad_sqr:", grad_sqr, self._gain_invalid)


        # reset backward call counters for next param update cycle
        self._last_final_backward_call = self._num_backward_calls = 0
        # Indicating backward is done.
        self._local_grad_sqr = None


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
        self.steps += 1
        assert self._local_grad_sqr is None, "Don't step without finishing backward phase"
        # Set original LR and set new LR.
        original_lr = []
        for pg_idx, param_group in enumerate(self._optimizer.param_groups):
            original_lr.append(param_group["lr"])
            param_group["lr"] = self.gain(pg_idx=pg_idx) * param_group["lr"]
        res = None
        # Step it.
        if self._scaler:
            # if self._gain_invalid[0] != 0 and self._model is not None:
            #     print("STEPPING WHEN PROBLEMATIC", norm)
            # FIXME: Google BERT uses grad norm clipping with Adam optimizer (missing in NV impl because it uses LAMB)
            self._scaler.unscale_(self._optimizer)
            norm = torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=5.0) #TODO: pass as config
            # if self._rank == 0: print("norm:", norm)
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


    def set_num_gradients_to_accumulate(self, num_gradients_to_accumulate: int, update_smoothing: bool = True,) -> None:
        """Set the number of gradients to accumulate to a new value.

        This is experimental. This could be called while training so that
        we can gradually increasing the steps between updates. Almost always,
        `set_scale` needs to be called to update the scale as well.

        TODO (min): need a way of determine how much to increase the step size?

        TODO (min): have both `set_scale` and `set_num_gradients_to_accumulate`
        is hard to use and easy to make mistake. I think it is better
        to specific a specify a `base_scale`. But more discussion is
        needed here.

        Args:
            num_gradients_to_accumulate (int):
                Number of gradients to accumulate (calls to backward) between
                each optimizer step
            update_smoothing (bool):
                Whether to update smoothing factor or not. Default: True.
        """
        assert self._local_grad_sqr is None, "Don't change num_grad_to_accum in backward"
        assert num_gradients_to_accumulate >= 1, f"Invalid value {num_gradients_to_accumulate}"
        self._num_grads_to_accum = num_gradients_to_accumulate
        if update_smoothing:
            # Set smoothing based on effective world_size rather than scale here,
            # since world_size determines the number of samples being averaged over
            # at every update.
            #
            # When effective world size is large enough, smoothing is probably
            # not needed, so the smoothing factor is 0.
            # TODO: smoothing function can be a callback - allows end user to change smoothing rules
            self._smoothing = max(1 - self._world_size * self._num_grads_to_accum / 1000, 0)


    def _calculate_preconditioner(self, pg_idx, param, where="local"):
        """
        From openai paper - One might also use preconditioned gradients, obtained for example by dividing gradient 
        components by the squareroot of the Adam optimizer’s [KB14] accumulated variances.
        in case of ADAM - note that averages won't be very useful until we have done 1/(1-beta2) batches, so we
        ignore batch size predictions initially
        Q. should we not precondition for the initial steps? How does this affect AdaScale stats??
        TODO: Investigate other preconditioners
        """
        if not self._use_preconditioner:
            return torch.ones_like(param, memory_format=torch.preserve_format)
        else:
            if not self._is_adaptive or param not in self._optimizer.state:
                return torch.ones_like(param, memory_format=torch.preserve_format)
            # get current state for param
            state = self._optimizer.state[param]
            # get param group settings
            group = self._optimizer.param_groups[pg_idx]
            _, beta2 = group['betas']
            step = group['step']
            exp_avg_sq = state["exp_avg_sq"] #.clone()
            eps = self._opt_param_group['eps'][pg_idx]
            bias_correction = 1 - beta2 ** step
            pinv = (exp_avg_sq / bias_correction).sqrt().add_(eps)
            return pinv
