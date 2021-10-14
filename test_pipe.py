import torch
import timm.models.vision_transformer
import timm.models.efficientnet
from transformers import BertModel, BertConfig
import argparse
import torch.nn as nn
from torch.nn import *
import dataclasses
import time
import os
import traceback
import sys
import copy
import random
import numpy as np
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
	hsize = 1024
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

def pipe_setup(model, infer_inputs, n_devices, model_name):
	modules = [module for module in model.modules() if default_block(module)]
	model = assign_chunks(modules, n_devices)	
	model = torch.distributed.pipeline.sync.Pipe(model, chunks=n_devices, checkpoint='except_last', deferred_batch_norm=False).eval()
	torch.cuda.synchronize()
	return model

def run_fsdp(n_devices, model_name):
	cmd = 'python -m torch.distributed.run --nproc_per_node=' + str(n_devices) + ' FSDP.py ' + str(model_name) + ' -v'
	args = list(cmd.split(' '))
	try:
		p = subprocess.Popen(args)
		outs, errs = p.communicate()
	except:
		traceback.print_exc(file=sys.stdout)
		print(args)
		quit()
	filename = model_name + str(n_devices) + '.txt'
	fileread = str(open(filename,'r').read())
	try:
		runtime = float(fileread)
	except:
		print(fileread)
	os.remove(filename)
	return runtime

def run_pipeline(n_devices, model_name):
	cmd = 'python pipey.py ' + str(model_name) + ' -v' + ' -n_devices ' + str(n_devices)
	args = list(cmd.split(' '))
	try:
		p = subprocess.Popen(args)
		outs, errs = p.communicate()
	except:
		if model_name == 'Linear':
			traceback.print_exc(file=sys.stdout)
			print(args)
			quit()
		else:
			print("Inference failed for", model_name)
	filename = model_name + str(n_devices) + '.txt'
	fileread = str(open(filename,'r').read())
	try:
		runtime = float(fileread)
	except:
		if model_name == 'Linear':
				print(fileread)
		runtime = float('nan')
	os.remove(filename)
	return runtime

def plot(runtimes):
	x = dict([(model, dict([(implementation, []) for implementation in runtimes.keys()])) for model in runtimes['native'].keys()])
	y = dict([(model, dict([(implementation, []) for implementation in runtimes.keys()])) for model in runtimes['native'].keys()])
	for model in runtimes['native'].keys():
		for implementation in runtimes.keys():
			for n_devices in runtimes['native']['Linear'].keys():
				runtime = float(runtimes[implementation][model][n_devices])
				if runtime != float('nan'):
					y[model][implementation].append(runtime)
					x[model][implementation].append(int(n_devices))
				else:
					continue
			plt.scatter(x, y, label='implementation')
		plt.legend()
		plt.xlabel('n_devices')
		plt.ylabel('Forward Pass time (ms)')
		plt.title(str(model) + " Foward Pass Scaling")
		plt.savefig(str(model) + '_scaling.png')
		plt.close()


def main():
	runtimes = dict((implementation, {'EF':{}, 'VT':{}, 'Linear':{}, 'hugface':{}}) for implementation in ['native', 'FSDP'])
	for implementation in ['native', 'FSDP']:
		print("Implementation:", implementation)
		for n_devices in range(1,int(torch.cuda.device_count())+1):				
			print("Testing", n_devices,"devices:")
			#Model Inits
			set_seed()
			for model_name in ['Linear', 'EF', 'VT', 'hugface']:
				#setup model parallel
				if implementation == 'native':
					runtimes[implementation][model_name][str(n_devices) + '_gpus'] = run_pipeline(n_devices, model_name)
				else:
					if n_devices == 1:
						runtimes[implementation][model_name][str(n_devices) + '_gpus'] = runtimes['native'][model_name][str(n_devices) + '_gpus']
					else:
						runtimes[implementation][model_name][str(n_devices) + '_gpus'] = run_fsdp(n_devices, model_name)					
			print()
			print('#'*25)
		#report it
		print("Runtimes in ms:")
		print(runtimes)
		plot(runtimes)




if __name__ == "__main__":
	main()