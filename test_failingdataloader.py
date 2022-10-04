import torch
import gc
import time
import signal
import unittest
import itertools
import warnings
import tempfile
from torch import multiprocessing as mp
from torch.utils.data import TensorDataset
from torch.utils.data._utils import MP_STATUS_CHECK_INTERVAL
from torch.utils.data.dataset import random_split
from torch._utils import ExceptionWrapper

import random
import numpy as np

def set_seed():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)


def test_sampler_reproducibility():
    data = torch.randn(100, 2, 3, 5)
    labels = torch.randperm(50).repeat(2)
    dataset = TensorDataset(data, labels)
    from torch.utils.data import RandomSampler, WeightedRandomSampler, SubsetRandomSampler
    weights = [0.1, 0.9, 0.4, 0.7, 3.0, 0.6]
    for fn in (
        lambda: RandomSampler(dataset, num_samples=5, replacement=True, generator=torch.Generator().manual_seed(42)),
        lambda: RandomSampler(dataset, replacement=False, generator=torch.Generator().manual_seed(42)),
        lambda: WeightedRandomSampler(weights, num_samples=5, replacement=True, generator=torch.Generator().manual_seed(42)),
        lambda: WeightedRandomSampler(weights, num_samples=5, replacement=False, generator=torch.Generator().manual_seed(42)),
        lambda: SubsetRandomSampler(range(10), generator=torch.Generator().manual_seed(42)),
    ):
        assert (list(fn()) == list(fn()))

    for sampler, replacement in (
        (RandomSampler(dataset, num_samples=5, replacement=True), True),
        (RandomSampler(dataset, replacement=False),False),
        (WeightedRandomSampler(weights, num_samples=5, replacement=True), True),
        (WeightedRandomSampler(weights, num_samples=5, replacement=False),False),
        (SubsetRandomSampler(range(10)), None),
    ):

        set_seed()
        l1 = list(sampler) + list(sampler)

        set_seed()
        l2 = list(sampler) + list(sampler)
        assert (l1 == l2), str(l1) + '    ' + str(l2) + ' ' + str(sampler)

        its = (iter(sampler), iter(sampler))
        ls = ([], [])
        for idx in range(len(sampler)):
            for i in range(2):
                if idx == 0:
                    set_seed()
                ls[i].append(next(its[i]))
        assert (ls[0] == ls[1]), str(ls[0]) + '    ' + str(ls[1]) + ' ' + str(sampler)

torch.use_deterministic_algorithms(True)
test_sampler_reproducibility()