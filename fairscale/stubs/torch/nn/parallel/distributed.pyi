# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from ..modules import Module
from typing import Any, Optional, TypeVar
from .common_types import _devices_t, _device_t

T_co = TypeVar('T_co', covariant=True)

def get_rank(group: Any): ...

class DistributedDataParallel(Module[T_co]):
    process_group: Any = ...
    dim: int = ...
    module: Module[T_co] = ...
    device_ids: _devices_t = ...
    output_device: _device_t = ...
    broadcast_buffers: bool = ...
    check_reduction: bool = ...
    broadcast_bucket_size: float = ...
    bucket_bytes_cap: float = ...

    # TODO type process_group once `distributed` module is stubbed
    def __init__(self, module: Module[T_co], device_ids: Optional[_devices_t] = ...,
                 output_device: Optional[_device_t] = ..., dim: int = ...,
                 broadcast_buffers: bool = ..., process_group: Optional[Any] = ..., bucket_cap_mb: float = ...,
                 check_reduction: bool = ...) -> None: ...

    def forward(self, *inputs: Any, **kwargs: Any) -> T_co: ...

    def __call__(self, *inputs: Any, **kwargs: Any) -> T_co: ...
