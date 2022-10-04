import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
def fwd_bwd(features, scaler, model, autocast, label, loss):
	if autocast:
		with torch.cuda.amp.autocast(dtype=torch.bfloat16):
			total_loss = loss(model(features), label)
			total_loss = total_loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
	else:
		total_loss = loss(model.bfloat16()(features.bfloat16()), label)
		total_loss = total_loss.mean() 
	scaler.scale(total_loss).backward()
	return total_loss


def train_one_step(model, optimizer, features, scaler, autocast, label, loss):
	total_loss = fwd_bwd(features, scaler, model, autocast, label, loss)
	scaler.step(optimizer)
	# optimizer.zero_grad(set_to_none=True)
	optimizer.zero_grad()
	scaler.update()
				

	return total_loss

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
	args = parser.parse_args()
	torch.cuda.set_device(args.local_rank)
	device = torch.device("cuda", args.local_rank)
	torch.distributed.init_process_group(backend="nccl")
	if args.local_rank == 0:
			print("BFloat16 nccl debug:")
	n_gpu = 2
	loss = torch.nn.CrossEntropyLoss().cuda()
	scaler = torch.cuda.amp.GradScaler(enabled=False)
	model = DDP(torch.nn.Linear(10,5).cuda())
	optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
	for i in range(10):
		features=torch.randn((5,10)).cuda()
		train_one_step(model, optimizer, features, scaler, False, torch.randint(2, (5,)).cuda(), loss)
	if args.local_rank == 0:
			print("\n"*5)
			print("AUTOCAST nccl debug:")
	scaler = torch.cuda.amp.GradScaler(enabled=True)
	loss = torch.nn.CrossEntropyLoss().cuda()
	model = DDP(torch.nn.Linear(10,5).cuda())
	optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
	for i in range(10):
		features=torch.randn((5,10)).cuda()
		train_one_step(model, optimizer, features, scaler, True, torch.randint(2, (5,)).cuda(), loss)


if __name__ == "__main__":
	main()

