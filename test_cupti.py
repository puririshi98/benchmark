import collections
import gc
import io
import json
import os
import unittest

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.testing._internal.common_cuda import TEST_MULTIGPU
from torch.testing._internal.common_utils import (
    TestCase, run_tests, TEST_WITH_ASAN, TEST_WITH_ROCM, IS_WINDOWS,
    TemporaryFileName, TemporaryDirectoryName)
from torch.autograd.profiler import profile as _profile
from torch.profiler import (
    kineto_available, profile, record_function, supported_activities,
    DeviceType, ProfilerAction, ProfilerActivity
)
def test_profiler_fwd_bwd_link():
	with _profile(use_kineto=True) as prof:
		t1, t2 = torch.ones(1, requires_grad=True), torch.ones(1, requires_grad=True)
		z = torch.add(t1, t2)
		y = torch.ones(1)
		loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
		loss.backward()
	with TemporaryFileName(mode="w+") as fname:
		prof.export_chrome_trace(fname)
		with io.open(fname, 'r') as f:
			j = json.load(f)
			
			events = j["traceEvents"]
			ts_to_name = {}
			flow_s_to_ts = {}
			flow_f_to_ts = {}
			for e in events:
				print(e)
				print("cat" in e and "name" in e and e["cat"] == "forward_backward" and e["name"] == "fwd_bwd")

				if e["ph"] == "X":
					ts_to_name[e["ts"]] = e["name"]
				if "cat" in e and "name" in e and e["cat"] == "forward_backward" and e["name"] == "fwd_bwd":
					print("enter iffy")
					if e["ph"] == "s":
						flow_s_to_ts[e["id"]] = e["ts"]
					elif e["ph"] == "f":
						flow_f_to_ts[e["id"]] = e["ts"]
			print(len(flow_s_to_ts))
			assert (len(flow_s_to_ts) == 2)
			assert (len(flow_f_to_ts) == 2)
			assert (1 in flow_s_to_ts.keys())
			assert (1 in flow_f_to_ts.keys())
			assert (2 in flow_s_to_ts.keys())
			assert (2 in flow_f_to_ts.keys())
			s_ts_1 = flow_s_to_ts[1]
			f_ts_1 = flow_f_to_ts[1]
			s_ts_2 = flow_s_to_ts[2]
			f_ts_2 = flow_f_to_ts[2]
			assert (all([ts in ts_to_name.keys() for ts in [s_ts_1, f_ts_1, s_ts_2, f_ts_2]]))
			assert (ts_to_name[s_ts_1] == "aten::binary_cross_entropy_with_logits")
			assert (ts_to_name[s_ts_2] == "aten::add")
test_profiler_fwd_bwd_link()