import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
​import torch.cuda.nvtx as nvtx
import math
class Fusion(nn.Module):
	def __init__(self):
		super(Fusion, self).__init__()
		self.batchnorm_size = ((1,24,112,112),(24,24,24,24))
		self.ptwise = (1,24,112,112)
		self.conv2d = ((1,24,112,112),(24,1,33))
		self.convwt = torch.randn(conv2d[0], device="cuda", dtype=torch.float)
​
	def forward(self, inputy):
		out1 = F.batch_norm(inputy, 1.5, 4.2, weight=None, bias=None, training=False) + inputy

		out2 = F.conv2d(out1,self.convwt)
		return out
# eager is 10 passes on data
# fuesd is 3
inner_dim = 197
if __name__ == "__main__" :
	torch.cuda.cudart().cudaProfilerStart()
	inputs = torch.randn((1, 24, 112,112), device="cuda", dtype=torch.float, requires_grad=False)
	model = Fusion()
	model.cuda()
	model.eval()
	nvtx.range_push("replaying eager")
	for idx in range(10) :
		out = model(inputs, mask_bool)
	nvtx.range_pop()
	​torch._C._jit_set_nvfuser_enabled(True)
	torch._C._jit_set_texpr_fuser_enabled(False)
	torch._C._jit_set_profiling_executor(True)
	torch._C._jit_set_profiling_mode(True)
	torch._C._jit_override_can_fuse_on_cpu(False)
	torch._C._jit_override_can_fuse_on_gpu(False)
	torch._C._jit_set_bailout_depth(20)
	jit_model = torch.jit.script(model)
	nvtx.range_push("replaying nvfuser")
	for idx in range(10) :
		out = jit_model(inputs, mask_bool)
	nvtx.range_pop()
	torch.cuda.cudart().cudaProfilerStop()