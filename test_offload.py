import torch
import argparse
import torch.nn as nn
import time
import os
import traceback
import sys
import copy
import random
import numpy as np
import subprocess


def gen_simple_linear_model(n):
	layer_list = []
	hsize = 1024
	for i in range(n):
		layer_list += [torch.nn.Linear(hsize,int(hsize))]
		if i != n_devices-1:
			layer_list += [torch.nn.ReLU()]
	return torch.nn.Sequential(*layer_list)


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

def run_offload(n, implementation):
	cmd = 'python3 ' + str(implementation) + '.py ' + str(n)
	args = list(cmd.split(' '))
	try:
		p = subprocess.Popen(args)
		outs, errs = p.communicate()
	except:
		traceback.print_exc(file=sys.stdout)
		print(args)
		print("Inference failed for", implementation)
	filename = implementation + str(n) + '.txt'
	fileread = str(open(filename,'r').read())
	runtime = float(fileread)
	os.remove(filename)
	return runtime


def plot(runtimes):
	import matplotlib.pyplot as plt
	x = dict([(implementation, []) for implementation in runtimes.keys()])
	y = dict([(implementation, []) for implementation in runtimes.keys()])
	for implementation in runtimes.keys():
		for n_layers in runtimes['native'].keys():
			try:
				runtime = float(runtimes[implementation][n_layers])
				y[model][implementation].append(runtime)
				[1024 * 1025 / 1000000 * [model][implementation].append(int(n_layers.split(' ')[0]))]
			except:
				continue
	plt.scatter(x[model][implementation], y[model][implementation], label=str(implementation))
	plt.legend()
	plt.xlabel('Parameters (Millions)')
	plt.ylabel('Forward Pass time (ms)')
	plt.title(str("Offload Foward Pass Scaling"))
	plt.savefig('offload_scaling.png')
	plt.close()


def main():
	# implementations = ['native', 'deepspeed']
	implementations = ['native']
	runtimes = dict((implementation, {}) for implementation in implementations)
	for implementation in implementations:
		print("Implementation:", implementation)
		for n in range(100,3200,100):
			print("Testing", n,"layers:")
			if i == 3100:
				torch.cuda.cudart().cudaProfilerStart()
			runtime = run_offload(n, implementation)
			runtimes[implementation][str(n)+' layers'] = runtime
			print(runtime)
			print('#'*10)
	#report it
	torch.cuda.cudart().cudaProfilerStop()
	print("Runtimes in ms:")
	print(runtimes)
	plot(runtimes)




if __name__ == "__main__":
	main()