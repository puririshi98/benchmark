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
models = [timm.create_model(name, pretrained=False, scriptable=True).cuda().float() for name in model_names]
torch.distributed.init_process_group("nccl", rank=args.local_rank, world_size=world)
for model, name in zip(models,model_names):	
	shapes = [param.size() for param in model.parameters()]
	sizes = [param.numel() for param in model.parameters()]
	if args.local_rank == 0:
		print("Param Shapes for",name+":",shapes)
		print("Param Sizes for",name+":",sizes)
	device = torch.device("cuda:%d" % args.local_rank)
	for shape in shapes:
		tensors = [torch.full(shape, args.local_rank + 1 + i, device=device, dtype=torch.float) for i in range(5)]
		torch.distributed.all_reduce_coalesced(tensors)
		for i, t in enumerate(tensors):
			assert torch.equal(t, torch.full_like(t, world * (i + (world + 1.) / 2.)))
if args.local_rank == 0:
	print("Passed Correctness test!")

if args.local_rank == 0:
	print("Running perf test...")
torch.cuda.cudart().cudaProfilerStart()
with torch.autograd.profiler.emit_nvtx(record_shapes=True):
	for model, name in zip(models, model_names):	
		nvtx.range_push("Profiling Model: " + str(name))
		shapes = [param.size() for param in model.parameters()]
		sizes = [param.numel() for param in model.parameters()]
		device = torch.device("cuda:%d" % args.local_rank)
		for shape in shapes:
			nvtx.range_push("Warmup!")
			for i in range(3):
				tensors = [torch.full(shape, args.local_rank + 1 + i, device=device, dtype=torch.float) for i in range(5)]
				torch.distributed.all_reduce_coalesced(tensors)
			nvtx.range_pop()
			tensors = [torch.full(shape, args.local_rank + 1 + i, device=device, dtype=torch.float) for i in range(5)]
			nvtx.range_push("Coalesce:" + str(np.prod(shape)))
			torch.distributed.all_reduce_coalesced(tensors)
			nvtx.range_pop()
			nvtx.range_push("Warmup!")
			for i in range(3):
				tensors = [torch.full(shape, args.local_rank + 1 + i, device=device, dtype=torch.float) for i in range(5)]
				torch.distributed.all_reduce(torch.cat(tensors))
			nvtx.range_pop()
			tensors = [torch.full(shape, args.local_rank + 1 + i, device=device, dtype=torch.float) for i in range(5)]
			nvtx.range_push("Flat All Reduce Size:" + str(np.prod(shape)))
			torch.distributed.all_reduce(torch.cat(tensors))
			nvtx.range_pop()
		nvtx.range_pop()
torch.cuda.cudart().cudaProfilerStop()
if args.local_rank == 0:
	print("Finished perf test!")