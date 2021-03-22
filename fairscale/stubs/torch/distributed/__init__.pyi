# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Any, List, Union, Optional
from torch import Tensor
import datetime

from . import rpc as rpc
from . import distributed_c10d as distributed_c10d

class Backend:
    GLOO: str
    MPI: str
    NCCL: str

class ProcessGroup:
    def size(self) -> int: ...
    def rank(self) -> int: ...

class ReduceOp:
    SUM: ReduceOp
    PRODUCT: ReduceOp
    MIN: ReduceOp
    MAX: ReduceOp
    BAND: ReduceOp
    BOR: ReduceOp
    BXOR: ReduceOp

def get_rank(group: Any = None) -> int: ...

def get_world_size(group: Any = None) -> int: ...
def get_backend(group: Optional[Any] = None) -> Any: ...
def broadcast(tensor: Tensor, src: Any, group: Any, async_op: Any = False): ...
def gather(tensor: Tensor, gather_list: Optional[List[Tensor]], dst: Any, group:Optional[ProcessGroup] = None, async_op: Optional[bool] = False): ...
def reduce(tensor: Tensor, dst: Any, op: Optional[Any]=ReduceOp.SUM, group:Optional[ProcessGroup] = None, async_op: Optional[bool] = False): ...
def broadcast_object_list(object_list: List[Any], src: int, group:Optional[ProcessGroup] = None): ...

def is_initialized() -> bool: ...

def init_process_group(backend: Union[str, Backend], init_method: Optional[str] = None, timeout: datetime.timedelta = datetime.timedelta(0, 1800), rank: Optional[int] = None, world_size: Optional[int] = None): ...
def new_group(ranks: List[int], timeout: datetime.timedelta = datetime.timedelta(0, 1800), backend: Union[None, str, Backend] = None): ...

def all_to_all(output: List[Tensor], input: List[Tensor], group:Optional[ProcessGroup] = None, async_op: bool = False): ...
def all_to_all_single(output: Tensor, input: Tensor, output_split_size: Optional[List[int]] = None, input_split_size: Optional[List[int]] = None, group:Optional[ProcessGroup] = None, async_op: bool = False): ...
def all_reduce(tensor: Tensor, op: ReduceOp = ReduceOp.SUM, group:Optional[ProcessGroup] = None, async_op: bool = False): ...
def all_gather(tensor_list: List[Tensor], tensor: Tensor, group:Optional[ProcessGroup] = None, async_op: bool = False): ...

def destroy_process_group() -> None: ...

def send(tensor: Tensor, dst: int, group: Optional[ProcessGroup] = None, tag: Optional[int] = None) -> None: ...
def isend(tensor: Tensor, dst: int, group: Optional[ProcessGroup] = None, tag: Optional[int] = None) -> None: ...
def recv(tensor: Tensor, src: Optional[int] = None, group: Optional[ProcessGroup] = None, tag: Optional[int] = None) -> int: ...
def irecv(tensor: Tensor, src: Optional[int] = None, group: Optional[ProcessGroup] = None, tag: Optional[int] = None) -> int: ...

class group(object):
    WORLD: Any

class RRef: ...