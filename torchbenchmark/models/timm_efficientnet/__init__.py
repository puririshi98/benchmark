# Generated by gen_timm_models.py
import torch
import timm.models.efficientnet

from ...util.model import BenchmarkModel
from torchbenchmark.tasks import COMPUTER_VISION
from .config import TimmConfig
import torch.cuda.nvtx as nvtx
class Model(BenchmarkModel):
	task = COMPUTER_VISION.CLASSIFICATION

	def __init__(self, device=None, jit=False, variant='mixnet_m', precision='float32'):
		super().__init__()
		self.device = device
		self.jit = jit
		self.model = timm.create_model(variant, pretrained=False, scriptable=True)
		self.cfg = TimmConfig(model = self.model, device = device, precision = precision)
		self.model.to(
			device=self.device,
			dtype=self.cfg.model_dtype
		)
		if device == 'cuda':
			torch.cuda.empty_cache()
		if jit:
			self.model = torch.jit.script(self.model)
			assert isinstance(self.model, torch.jit.ScriptModule)

	def _gen_target(self, batch_size):
		return torch.empty(
			(batch_size,) + self.cfg.target_shape,
			device=self.device, dtype=torch.long).random_(self.cfg.num_classes)

	def _step_train(self):
		nvtx.range_push('zeroing optimizer grad')
		self.cfg.optimizer.zero_grad()
		nvtx.range_pop()
		nvtx.range_push('forward pass')
		output = self.model(self.cfg.example_inputs)
		nvtx.range_pop()
		if isinstance(output, tuple):
			output = output[0]
		nvtx.range_push('__gen__target')
		target = self._gen_target(output.shape[0])
		nvtx.range_pop()
		nvtx.range_push('loss')
		self.cfg.loss(output, target).backward()
		nvtx.range_pop()
		nvtx.range_push('step')
		self.cfg.optimizer.step()
		nvtx.range_pop()


	def _step_eval(self):
		nvtx.range_push('eval')
		output = self.model(self.cfg.infer_example_inputs.half())
		nvtx.range_pop()

	def get_module(self):
		return self.model, (self.cfg.example_inputs,)

	def train(self, niter=1):
		self.model.train()
		graphs=True
		if graphs:
			self.model.to(memory_format=torch.channels_last)
			niter = 8
			s = torch.cuda.Stream()
			torch.cuda.synchronize()
			with torch.cuda.stream(s):
				nvtx.range_push('warming up')
				for _ in range(5):
					self._step_train()
				nvtx.range_pop()
				torch.cuda.empty_cache()
				g = torch.cuda._Graph()
				torch.cuda.synchronize()
				nvtx.range_push('capturing graph')
				g.capture_begin()
				self._step_train()
				g.capture_end()
				nvtx.range_pop()
				torch.cuda.synchronize()
			nvtx.range_push('replaying')
			for _ in range(niter-3):
				g.replay()
				torch.cuda.synchronize()
			nvtx.range_pop()
		else:
			for _ in range(niter):
				self._step_train()
	# TODO: use pretrained model weights, assuming the pretrained model is in .data/ dir
	def eval(self, niter=1):
		torch.cuda.cudart().cudaProfilerStart()
		with torch.autograd.profiler.emit_nvtx(record_shape=True):
			self.model.eval()
			torch.backends.cudnn.benchmark = True
			with torch.no_grad():
				self.model = self.model.half()
				graphs=True
				if graphs:
					niter = 8
					s = torch.cuda.Stream()
					torch.cuda.synchronize()
					with torch.cuda.stream(s):
						nvtx.range_push('warming up')
						print('warming up')
						for _ in range(5):
							self._step_eval()
						nvtx.range_pop()
						torch.cuda.empty_cache()
						g = torch.cuda._Graph()
						torch.cuda.synchronize()
						nvtx.range_push('capturing graph')
						print('capturing graph')
						g.capture_begin()
						self._step_eval()
						g.capture_end()
						nvtx.range_pop()
						torch.cuda.synchronize()
					nvtx.range_push('replaying')
					print('replaying')
					for _ in range(niter-3):
						g.replay()
						torch.cuda.synchronize()
					nvtx.range_pop()
				else:
					for _ in range(niter):
						self._step_eval()
		torch.cuda.cudart().cudaProfilerStop()
if __name__ == "__main__":
	for device in ['cpu', 'cuda']:
		for jit in [False, True]:
			print("Test config: device %s, JIT %s" % (device, jit))
			m = Model(device=device, jit=jit)
			m, example_inputs = m.get_module()
			m(example_inputs)
			m.train()
			m.eval()