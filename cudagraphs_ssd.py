import torch
precision = 'fp16'
model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math=precision)
niter = 8
torch.cuda.cudart().cudaProfilerStart()
with torch.autograd.profiler.emit_nvtx(record_shapes=True):
	model.eval().cuda()
	torch.backends.cudnn.benchmark = True
	with torch.no_grad():
		if precision == 'fp16':
			self.model = self.model.half()
		s = torch.cuda.Stream()
		torch.cuda.synchronize()
		with torch.cuda.stream(s):
			nvtx.range_push('warming up')
			print('warming up')
			for _ in range(5):
				self._step_eval(precision)
			nvtx.range_pop()
			torch.cuda.empty_cache()
			g = torch.cuda._Graph()
			torch.cuda.synchronize()
			nvtx.range_push('capturing graph')
			print('capturing graph')
			g.capture_begin()
			self._step_eval(precision)
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
