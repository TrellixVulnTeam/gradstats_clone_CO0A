import math
from typing import TypeVar, Optional, Iterator

import torch
from torch.utils.data.distributed import DistributedSampler, Sampler, Dataset
import torch.distributed as dist


T_co = TypeVar('T_co', covariant=True)


class ReplacementDistributedSampler(DistributedSampler):
    r"""
    DataSampler used for adascale with sampling replacement.
    .. note::
        Dataset is assumed to be of constant size.
    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.
    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.
    Example::
        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            # self.num_samples = math.ceil(
                # `type:ignore` is required because Dataset cannot provide a default __len__
                # see NOTE in pytorch/torch/utils/data/sampler.py
            #     (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            # )
            self.num_samples = len(self.dataset) - self.num_replicas
        else:
            # self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
            self.num_samples = len(self.dataset)
        self.total_size = self.num_samples
        self.shuffle = shuffle


    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch, seed and rank
            g = torch.Generator()
            import random
            #g.manual_seed(self.seed + self.rank + random.randint(0, 2 ** 32 - 1))
            seed = random.randint(0, 2 ** 32 - 1)
            g.manual_seed(seed)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
            # print(f"epoch is {self.epoch}, seed is {self.seed}, rank is {self.rank}, \n"
            #       f"dataset total_size is {self.total_size}, indices len is {len(indices)}, \n"
            #       f"first 10 indices of current dataloader are \n "
            #       f"{indices[:10]}")
            print(f'DEBUG ReplacementDistributedSampler iter, rank:{self.rank}, seed:{seed}, epoch:{self.epoch} '
                  f'indices: {indices[:10]}')
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        # indices = indices[self.rank:self.total_size:self.num_replicas]
        # import random
        # random.seed(self.rank)
        # sample subset but keep the shuffled order
        #index_list = random.sample(range(len(indices)), self.total_size // self.num_replicas)
        #indices = [indices[i] for i in sorted(index_list)]
        #print(indices[:10])
        # assert len(indices) == self.num_samples
        return iter(indices)

