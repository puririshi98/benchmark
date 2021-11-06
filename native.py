import torch
import argparse
from test_offload import set_seed, gen_simple_linear_model
import traceback
import time
import os
import sys

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("N", type=int, default=1, help="Number of in_feat=1024, out_feat=1024 Linear Layers to connect with ReLUs")
	args = parser.parse_args()
	set_seed()
	n=args.N
	model = gen_simple_linear_model(n)
	implementation = str(__file__).split(os.sep)[-1].split('.')[0]
	infer_inputs = (torch.randn((64,1024)).cuda(),)
	with open(implementation + str(n) + '.txt','w+') as f:
		model = model.cuda().eval()
		try:
			with torch.autograd.graph.save_on_cpu():
				with torch.cuda.amp.autocast():
					since = time.time()
					for i in range(100):
						model(*infer_inputs)
			runtime = str(round((time.time()-since)*10, 2))
			print(runtime, file=f)
		except Exception as e:
			print("On", n, "devices", file=f)
			print("Inference Failed for:", implementation, file=f)
			traceback.print_exc(file=f)


if __name__ == "__main__":
	main()