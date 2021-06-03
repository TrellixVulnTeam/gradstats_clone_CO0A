# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2018-2019 NVIDIA CORPORATION. All rights reserved.
import datetime
import logging
import time
import math

import torch
import torch.distributed as dist

from maskrcnn_benchmark.utils.comm import get_world_size, is_main_process, synchronize
from maskrcnn_benchmark.utils.metric_logger import MetricLogger

from apex import amp
from torch.nn.utils import clip_grad_norm_, clip_grad_value_


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses

# Instead of zeroing, set parameter grads to None
# Prevents extraneous copy as we're not accumulating
def set_grads_to_none(model):
    for param in model.parameters():
        param.grad = None


def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
    disable_allreduce_for_logging,
    iters_per_epoch,
    per_iter_start_callback_fn=None,
    per_iter_end_callback_fn=None,
    scale=1.0,
    clip_grad_norm = 0.0,
    clip_grad_val = 0.0,
    use_adascale = False,
    measure_gns = False
):
    assert not (clip_grad_norm > 0.0 and clip_grad_val > 0.0), "Can't set both clip norm and clip value"
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    if use_adascale or measure_gns:
        # this wrapper has moved to make_optimizer - this has to be done before amp.initialize is called
        # scales the variance stats accordingly
        optimizer.set_scale(scale)

    def prefetcher(load_iterator):
        prefetch_stream = torch.cuda.Stream()
        pad_batches = []

        def _prefetch():
            try:
                # I'm not sure why the trailing _ is necessary but the reference used
                # "for i, (images, targets, _) in enumerate(data_loader):" so I'll keep it.
                images, targets, _ = next(load_iterator)
            except StopIteration:
                return None, None

            with torch.cuda.stream(prefetch_stream):
                # TODO:  I'm not sure if the dataloader knows how to pin the targets' datatype.
                targets = [target.to(device, non_blocking=True) for target in targets]
                images = images.to(device, non_blocking=True)

            return images, targets

        next_images, next_targets = _prefetch()

        while next_images is not None:
            torch.cuda.current_stream().wait_stream(prefetch_stream)
            current_images, current_targets = next_images, next_targets
            next_images, next_targets = _prefetch()
            yield current_images, current_targets

    synchronize()
    optimizer.zero_grad()
    step = 0 # adascale specific
    epoch = 0
    for iteration, (images, targets) in enumerate(prefetcher(iter(data_loader)), start_iter):
        if iteration // iters_per_epoch > epoch:
            epoch += 1
            data_loader.batch_sampler.batch_sampler.sampler.set_epoch(epoch)
        if per_iter_start_callback_fn is not None:
            per_iter_start_callback_fn(iteration=iteration)

        data_time = time.time() - end
        # iteration = iteration + 1
        arguments["iteration"] = iteration


        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        if not disable_allreduce_for_logging:
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss=losses_reduced, **loss_dict_reduced)
        else:
            meters.update(loss=losses, **loss_dict)

        with amp.scale_loss(losses, optimizer) as scaled_losses:
            scaled_losses.backward()

        
        # TODO: What is a good value for clipping - CHECK Adaptive Gradient Clipping - Ideally this should be inside AdaScale wrapper and not in user code
        if clip_grad_norm > 0.0:
            clip_grad_norm_(amp.master_params(optimizer), clip_grad_norm) # 1.5 mrcnn novo
        elif clip_grad_val > 0.0:
            clip_grad_value_(amp.master_params(optimizer), clip_grad_val) # 0.5 mrcnn sgdw 

        if measure_gns:
            gns = optimizer.gns(scale_one_batch_size=32) #FIXME: pass from main training loop
        else:
            gns = 0.0

        if use_adascale:
            gain = optimizer.scale_invariant_steps(aggressive_base_schedule=step > 500) # adascale specific FIXME: hardcoded for now
            prev_steps = math.floor(step)
            step = step + gain
            new_steps = math.floor(step)
            scale_invariant_steps = new_steps - prev_steps
            # make adascale step (note that in current implementation this corresponds to gain which may not be same as scale invariant steps, e.g. in NovoGrad case)
            optimizer.step()
            # set_grads_to_none(model)
            optimizer.zero_grad()
            #FIXME: better interface is scheduler.set_step(curr_effective_step)
            for _ in range(scale_invariant_steps):
                scheduler.step() # current scheduler is step based
        else:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            if not use_adascale:
                # space fillers for logs
                gain = 1.0
                step = iteration

            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                        "gain: {gain:.6f}",
                        "gns: {gns:.6f}",
                        "scale: {scale}",
                        "effective_lr: {elr:.6f}",
                        "scale_invariant_steps: {step}"
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    gain=gain,
                    gns=gns,
                    scale=scale,
                    elr=optimizer.param_groups[0]["lr"] * gain,
                    step=step
                )
            )
        if iteration % checkpoint_period == 0 and arguments["save_checkpoints"]:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if step >= max_iter and arguments["save_checkpoints"]:
            checkpointer.save("model_final", **arguments)

        # per-epoch work (testing)
        if per_iter_end_callback_fn is not None:
            # Note: iteration has been incremented previously for
            # human-readable checkpoint names (i.e. 60000 instead of 59999)
            # so need to adjust again here
            early_exit = per_iter_end_callback_fn(iteration=iteration-1)
            if early_exit:
                break

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
    if per_iter_end_callback_fn is not None:
        if early_exit:
            return True
        else:
            return False
    else:
        return None

