import torch
import functools
import warnings
import collections
try:
	import numpy as np
	HAS_NUMPY = True
except ModuleNotFoundError:
	HAS_NUMPY = False
from torch._six import string_classes


