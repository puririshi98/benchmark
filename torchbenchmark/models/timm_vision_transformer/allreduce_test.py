# Generated by gen_timm_models.py
import torch
import timm.models.vision_transformer
import timm.models.efficientnet
import time
import argparse
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--local_rank",type=int)
args = parser.parse_args()
device='cuda'
precision='float32'
model_names = ['vit_small_patch16_224', 'mixnet_m']
models = [timm.create_model(name, pretrained=False, scriptable=True).cuda().float() for name in model_names]
dist.init_process_group("nccl", rank=args.local_rank, world_size=world)
for model, name in zip(models,model_names):	
	shapes = [param.size() for param in model.parameters()]
	sizes = [param.numel() for param in model.parameters()]
	print("Param Shapes for",name+":",shapes)
	print("Param Sizes for",name+":",sizes)
	world=4
	device = torch.device("cuda:%d" % args.local_rank)
	for shape in shapes:
		tensors = [torch.full(shape, args.local_rank + 1 + i, device=device, dtype=torch.float) for i in range(5)]
		torch.distributed.all_reduce_coalesced(tensors)
		for i, t in enumerate(tensors):
			assert torch.equal(t, torch.full_like(t, world * (i + (world + 1.) / 2.)))


