import torch
import timm.models.vision_transformer
import timm.models.efficientnet
from transformers import BertModel, BertConfig
import argparse
import torch.nn as nn
from torch.nn import *
import dataclasses
import time
import torch.distributed.pipeline
import torch.distributed.pipeline.sync
import os
import traceback
import sys
import copy
import random
import numpy as np
import torch.distributed.rpc as rpc
import shlex
import subprocess

def resolve_precision(precision: str):
	assert precision in ('amp', 'float16', 'bfloat16', 'float32')
	use_amp = False
	model_dtype = torch.float32
	data_dtype = torch.float32
	if precision == 'amp':
		use_amp = True
	elif precision == 'float16':
		model_dtype = torch.float16
		data_dtype = torch.float16
	elif precision == 'bfloat16':
		model_dtype = torch.bfloat16
		data_dtype = torch.bfloat16
	return use_amp, model_dtype, data_dtype

@dataclasses.dataclass
class OptimizerOption:
	lr: float
	opt: str
	weight_decay: float
	momentum: float

class TimmConfigEF:
	def _init_input(self):
		self.example_inputs = torch.randn(
			(self.batch_size,) + self.input_size, dtype=self.data_dtype)
		self.infer_example_inputs = torch.randn(
			(self.batch_size,) + self.input_size, dtype=self.data_dtype)

	def __init__(self, model, precision):
		self.model = model
		self.use_amp, self.model_dtype, self.data_dtype = resolve_precision(precision)
		# Configurations
		self.batch_size = 64
		self.num_classes = self.model.num_classes
		self.target_shape = tuple()
		self.input_size = self.model.default_cfg["input_size"]
		self._init_input()

class TimmConfigVT:
	def _init_input(self):
		self.example_inputs = torch.randn(
			(self.batch_size,) + self.input_size, dtype=self.data_dtype)
		self.infer_example_inputs = torch.randn(
			(self.batch_size,) + self.input_size, dtype=self.data_dtype)

	def __init__(self, model, precision):
		self.model = model
		self.use_amp, self.model_dtype, self.data_dtype = resolve_precision(precision)
		# Configurations
		self.batch_size = 64
		self.num_classes = self.model.num_classes
		self.target_shape = tuple()
		self.input_size = self.model.default_cfg["input_size"]
		self._init_input()

def set_seed():
	random.seed(0)
	np.random.seed(0)
	torch.manual_seed(0)
	torch.cuda.manual_seed_all(0)

def gen_simple_linear_model(n_devices):
	layer_list = []
	hsize = 1024*8
	for i in range(n_devices):
		layer_list += [torch.nn.Linear(hsize,int(hsize))]
		if i != n_devices-1:
			layer_list += [torch.nn.ReLU()]
	return torch.nn.Sequential(*layer_list)

def default_block(module):
	types = [i for i in list(dir(torch.nn)) if i!='Module' and i!='Sequential' and i!='ModuleList' and any(x.isupper() for x in i) and '_' not in i]
	return str(module.__class__.__name__) in types

def assign_chunks(modules, n_devices):
	num_modules = len(modules)
	new_Module = torch.nn.Sequential(*modules)
	modules = [module for module in new_Module.modules() if default_block(module)]
	# print(modules)
	modules_in_each_chunk = int(num_modules / n_devices)
	start_ptr = 0
	for chunk in range(n_devices):
		if chunk == n_devices - 1:
			chunks = modules[start_ptr:]
		else:
			chunks = modules[start_ptr:(start_ptr+modules_in_each_chunk)]
		for module_x in chunks:
			module_x.cuda(chunk)
		start_ptr += modules_in_each_chunk
	return new_Module

def pipe_setup(model, ogmodel, infer_inputs, n_devices, model_name):
	modules = [module for module in model.modules() if default_block(module)]
	model = assign_chunks(modules, n_devices)	
	model = torch.distributed.pipeline.sync.Pipe(model, chunks=n_devices, checkpoint='except_last', deferred_batch_norm=False).eval()
	assert_msg = "pipelining for " + str(model_name) + ' damages correctness of the model'
	torch.cuda.synchronize()
	# assert torch.allclose(ogmodel(*infer_inputs), model(*infer_inputs), atol=1e-2), assert_msg
	return model

def run_fsdp(n_devices, model_name, verbose=False):
	cmd = 'python -m torch.distributed.launch --nproc_per_node=' + str(n_devices) + ' FSDP.py ' + str(model_name) + ' -v' if verbose else ''
	args = cmd.split(' ')
	p = subprocess.Popen(args)
	outs, errs = p.communicate()
	filename = model_name + str(n_devices) + '.txt'
	fileread = str(open(filename,'r').read())
	try:
		runtime = float(fileread)
	except:
		if verbose:
			print(fileread)
	os.remove(filename)
	return runtime

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-v", action='store_true', default=False, help="Verbose")
	args = parser.parse_args()
	os.environ['MASTER_ADDR'] = 'localhost'
	os.environ['MASTER_PORT'] = '29500'
	rpc.init_rpc('worker', rank=0, world_size=1)
	runtimes = dict((implementation, {'EF':{}, 'VT':{}, 'Linear':{}, 'hugface':{}}) for implementation in ['native', 'FSDP'])
	for implementation in ['native', 'FSDP']:
		print("Implementation:", implementation)
		for n_devices in range(1,int(torch.cuda.device_count())+1):
			if n_devices == 1 and implementation == 'FSDP':
				rpc.shutdown()
			print("Testing", n_devices,"devices:")
			#Model Inits
			set_seed()
			models = {'Linear':gen_simple_linear_model(n_devices),
				'EF':timm.create_model('mixnet_m', pretrained=False, scriptable=True),
				'VT':timm.create_model('vit_small_patch16_224', pretrained=False, scriptable=True),
				'hugface': BertModel(BertConfig())}
			set_seed()
			ogmodels = {'Linear':gen_simple_linear_model(n_devices),
				'EF':timm.create_model('mixnet_m', pretrained=False, scriptable=True),
				'VT':timm.create_model('vit_small_patch16_224', pretrained=False, scriptable=True),
				'hugface': BertModel(BertConfig())}
		
			for model_name in models.keys():
				#Model Setup
				model = models[model_name]
				ogmodel = ogmodels[model_name].cuda().eval()
				if model_name == 'EF':
					cfg = TimmConfigEF(model=model, precision='float32')
					infer_inputs = (cfg.infer_example_inputs.cuda(),)
				elif model_name == 'VT':
					cfg = TimmConfigVT(model=model, precision='float32')
					infer_inputs = (cfg.infer_example_inputs.cuda(),)
				elif model_name == 'Linear':
					infer_inputs = (torch.randn((64,1024*8)).cuda(),)
				elif model_name == 'hugface':
					vocab_size = 30522
					batchsize = 64
					seqlen = 128
					infer_inputs = (torch.randint(low=0, high=vocab_size, size=(batchsize, seqlen)).long().cuda(),)
				else:
					print("Model Not supported:", model_name)

				#setup model parallel
				if implementation == 'native':
					if n_devices > 1:
						try:
							model = pipe_setup(model, ogmodel, infer_inputs, n_devices, model_name)
						except Exception as e:
							print("On", n_devices, "devices")
							print("Could Not Succesfully Breakup:", model_name)
							print("With implementation:", implementation)
							if args.v:
								traceback.print_exc(file=sys.stdout)
							continue
					elif n_devices == 1:
						model =  model.cuda().eval()
					try:
						with torch.cuda.amp.autocast():
							since = time.time()
							for i in range(100):
								model(*infer_inputs)
							runtimes[implementation][model_name][str(n_devices) + '_gpus'] = str(round((time.time()-since)*10, 2)) + ' ms'
					except Exception as e:
						print("On", n_devices, "devices")
						print("Inference Failed for:", model_name)
						if args.v:
							traceback.print_exc(file=sys.stdout)
				else:
					if n_devices == 1:
						runtimes[implementation][model_name][str(n_devices) + '_gpus'] = runtimes['native'][model_name][str(n_devices) + '_gpus']
						continue
					runtimes[implementation][model_name][str(n_devices) + '_gpus'] = run_fsdp(n_devices, model_name, verbose=args.v)					
			print()
			print('#'*25)
	#report it
		print(runtimes)




if __name__ == "__main__":
	main()