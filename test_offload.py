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
	plt.scatter(runtimes[implementation].keys(), runtimes[implementation].values(), label=str(implementation))
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
		for n in range(500,10000,500):
			runtime = run_offload(n, implementation)
			runtimes[implementation][1024 * 1025 / 1000000 * n] = runtime
			print(runtimes)
	plot(runtimes)




if __name__ == "__main__":
	main()