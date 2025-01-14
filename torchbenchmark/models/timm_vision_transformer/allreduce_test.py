# Generated by gen_timm_models.py
import torch
import timm.models.vision_transformer
import timm.models.efficientnet
import time
import argparse
import torch.cuda.nvtx as nvtx

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--local_rank",type=int)
args = parser.parse_args()
device='cuda'
precision='float32'
model_names = ['vit_small_patch16_224', 'mixnet_m']
world=8
models = [timm.create_model(name, pretrained=False, scriptable=True).cuda().half() for name in model_names]
torch.distributed.init_process_group("nccl", rank=args.local_rank, world_size=world)
for model, name in zip(models,model_names):	
	shapes = [param.size() for param in model.parameters()]
	if args.local_rank == 0:
		print("Param Shapes for", name)
		for i, param in enumerate(model.parameters()):
			print(str(i) + "," + str(param.numel())  + ',' + str(param.dtype).split('.')[-1])
	device = torch.device("cuda:%d" % args.local_rank)
	for shape in shapes:
		tensors = [torch.full(shape, args.local_rank + 1 + i, device=device, dtype=torch.half) for i in range(5)]
		torch.distributed.all_reduce_coalesced(tensors)
		for i, t in enumerate(tensors):
			assert torch.equal(t, torch.full_like(t, world * (i + (world + 1.) / 2.)))
if args.local_rank == 0:
	print("Passed Correctness test!")

# if args.local_rank == 0:
# 	print("Running per param perf test...")
# torch.cuda.cudart().cudaProfilerStart()
# for model, name in zip(models, model_names):	
# 	nvtx.range_push("Profiling Model: " + str(name))
# 	shapes = [param.size() for param in model.parameters()]
# 	sizes = [param.numel() for param in model.parameters()]
# 	device = torch.device("cuda:%d" % args.local_rank)
# 	for shape in shapes:
# 		nvtx.range_push("Warmup!")
# 		tensors = [torch.full(shape, args.local_rank + 1 + i, device=device, dtype=torch.half) for i in range(5)]
# 		torch.distributed.all_reduce_coalesced(tensors)
# 		nvtx.range_pop()
# 		torch.cuda.synchronize()
# 		nvtx.range_push("Coalesce:" + str(torch.prod(torch.tensor(shape))))
# 		for i in range(10):
# 			torch.distributed.all_reduce_coalesced(tensors)
# 		nvtx.range_pop()
# 		nvtx.range_push("Warmup!")
# 		tensors = [torch.full(shape, args.local_rank + 1 + i, device=device, dtype=torch.half) for i in range(5)]
# 		torch.distributed.all_reduce(torch.cat(tensors))
# 		nvtx.range_pop()
# 		torch.cuda.synchronize()
# 		cats = torch.cat(tensors)
# 		nvtx.range_push("Flat All Reduce Size:" + str(torch.prod(torch.tensor(shape))))
# 		for i in range(10):
# 			torch.distributed.all_reduce(cats)
# 		nvtx.range_pop()
# 	nvtx.range_pop()
# torch.cuda.cudart().cudaProfilerStop()
# if args.local_rank == 0:
# 	print("Finished per param perf test!")

if args.local_rank == 0:
	print("Running perf test...")
torch.cuda.cudart().cudaProfilerStart()
for model, name in zip(models, model_names):	
	nvtx.range_push("Profiling Model: " + str(name))
	shapes = [param.size() for param in model.parameters()]
	device = torch.device("cuda:%d" % args.local_rank)
	nvtx.range_push("Warmup!")
	tensors = [torch.full(shape, args.local_rank + 1 + i, device=device, dtype=torch.half) for shape in shapes]
	torch.distributed.all_reduce_coalesced(tensors)
	nvtx.range_pop()
	torch.cuda.synchronize()
	nvtx.range_push("Coalesce:" + str(torch.prod(torch.tensor(shape))))
	for i in range(10):
		torch.distributed.all_reduce_coalesced(tensors)
	nvtx.range_pop()
	
	tensors = [torch.full(shape, args.local_rank + 1 + i, device=device, dtype=torch.half).reshape(-1) for shape in shapes]
	cats = torch.cat(tensors)
	nvtx.range_push("Warmup!")
	torch.distributed.all_reduce(cats)
	nvtx.range_pop()
	torch.cuda.synchronize()
	nvtx.range_push("Flat All Reduce Size:" + str(torch.prod(torch.tensor(shape))))
	for i in range(10):
		torch.distributed.all_reduce(cats)
	nvtx.range_pop()

	nvtx.range_pop() #Model name range
torch.cuda.cudart().cudaProfilerStop()
if args.local_rank == 0:
	print("Finished perf test!")