import torch
import torch.cuda.nvtx as nvtx
def _step_eval(model, batch):
	nvtx.range_push('eval')
	output = model(batch)
	nvtx.range_pop()
precision = 'fp16'
model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math=precision).half()
niter = 8
batch = torch.zeros(size=(1, 3, 300, 300)).cuda().half()
torch.cuda.cudart().cudaProfilerStart()
with torch.autograd.profiler.emit_nvtx(record_shapes=True):
	model.eval().cuda()
	torch.backends.cudnn.benchmark = True
	with torch.no_grad():
		s = torch.cuda.Stream()
		torch.cuda.synchronize()
		with torch.cuda.stream(s):
			nvtx.range_push('warming up')
			print('warming up')
			for _ in range(5):
				_step_eval(model, batch)
			nvtx.range_pop()
			torch.cuda.empty_cache()
			g = torch.cuda._Graph()
			torch.cuda.synchronize()
			nvtx.range_push('capturing graph')
			print('capturing graph')
			g.capture_begin()
			_step_eval(model, batch)
			g.capture_end()
			nvtx.range_pop()
			torch.cuda.synchronize()
		nvtx.range_push('replaying')
		print('replaying')
		for _ in range(niter-3):
			g.replay()
			torch.cuda.synchronize()
		nvtx.range_pop()
torch.cuda.cudart().cudaProfilerStop()
