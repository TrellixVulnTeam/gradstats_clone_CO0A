# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2018-2019 NVIDIA CORPORATION. All rights reserved.
import torch
import apex

from .lr_scheduler import WarmupMultiStepLR
from .cosine_lr_scheduler import CosineAnnealingWarmUpRestarts

from .fused_sgd import FusedSGD
from .fused_novograd import FusedNovoGrad
from .unfused_novograd import Novograd
from fairscale.optim import AdaScale
from .sgdw import SGDW

#for debugging only
from maskrcnn_benchmark.utils.comm import get_rank

def make_optimizer(cfg, model, use_adascale=False, enable_gns=False):
    params = []
    lr = cfg.SOLVER.BASE_LR
    weight_decay = cfg.SOLVER.WEIGHT_DECAY

    use_sqrt = False
    bias_params = []
    bias_lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
    bias_weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS

    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        if "bias" in key:
            bias_params.append(value)
        else:
            params.append(value)

    is_fp16 = (cfg.DTYPE == "float16")
    if is_fp16: # with FP16_Optimizer wrapper
        if cfg.SOLVER.OPTIMIZER == "NovoGrad":
            optimizer = FusedNovoGrad(
                [
                    {"params": params, "lr": lr, "weight_decay": weight_decay},
                    {"params": bias_params, "lr": bias_lr, "weight_decay": bias_weight_decay}
                ],
                lr, betas=(cfg.SOLVER.BETA1, cfg.SOLVER.BETA2), eps=1e-7, grad_averaging=False, init_zero=False, reg_inside_moment=True, bias_correction=True)
        elif cfg.SOLVER.OPTIMIZER == "UnfusedNovoGrad":
            optimizer = Novograd(
                [
                    {"params": params, "lr": lr, "weight_decay": weight_decay},
                    {"params": bias_params, "lr": bias_lr, "weight_decay": bias_weight_decay}
                ],
                lr=lr, betas=(cfg.SOLVER.BETA1, cfg.SOLVER.BETA2), eps=1e-7, grad_averaging=False, weight_decay=weight_decay, luc=False)
        elif cfg.SOLVER.OPTIMIZER == "SGDW":
            use_sqrt = False
            optimizer = SGDW(
                [
                    {"params": params, "lr": lr, "weight_decay": weight_decay},
                    {"params": bias_params, "lr": bias_lr, "weight_decay": bias_weight_decay}
                ],
                lr, momentum=cfg.SOLVER.MOMENTUM)
        else:
            raise NotImplementedError
    else: # without FP16_Optimizer wrapper
        optimizer = apex.optimizers.FusedSGD(
            [
                {"params": params, "lr": lr, "weight_decay": weight_decay},
                {"params": bias_params, "lr": bias_lr, "weight_decay": bias_weight_decay}
            ],
            lr, momentum=cfg.SOLVER.MOMENTUM)

    if use_adascale or enable_gns:
        smoothing = None # auto adjust for current training scale
        if cfg.SOLVER.GNS_SMOOTHING > 0.0:
            smoothing = cfg.SOLVER.GNS_SMOOTHING
            assert smoothing <= 1.0, "Smoothing should be a positive float less than 1.0"
        optimizer = AdaScale(optimizer,
                             rank=get_rank(),
                             is_adaptive=(cfg.SOLVER.OPTIMIZER != "SGD" and cfg.SOLVER.OPTIMIZER != "SGDW"),
                             smoothing=smoothing,
                             summary_writer=None) #TODO: pass writer for detailed grad stats

    return optimizer


def make_lr_scheduler(cfg, optimizer):
    if cfg.SOLVER.LR_SCHEDULE == "COSINE":
        return CosineAnnealingWarmUpRestarts(
            optimizer,
            T_0 = cfg.SOLVER.MAX_ITER, # total steps solver.max_iter
            eta_max = cfg.SOLVER.BASE_LR, # max lr or base lr init_lr
            alpha = 0.001,
            gamma = 0.01,
            T_up = cfg.SOLVER.WARMUP_ITERS, # warmup steps  , warmupsteps
        )
    elif cfg.SOLVER.LR_SCHEDULE == "MULTISTEP":
        return WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.STEPS,
            cfg.SOLVER.GAMMA,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )

