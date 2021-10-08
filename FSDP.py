import fairscale
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
import torch
import timm.models.vision_transformer
import timm.models.efficientnet
from transformers import BertModel, BertConfig
import argparse
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("MODEL_NAME", type=str,  help="model name")
	parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
	args = parser.parse_args()
	torch.cuda.set_device(args.local_rank)
	device = torch.device("cuda", args.local_rank)
	torch.distributed.init_process_group(backend="nccl")
	n_gpu = torch.distributed.get_world_size()
	set_seed()
	model_name = args.MODEL_NAME
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
	model = FSDP(model.cuda()).eval()
	try:
		with torch.cuda.amp.autocast():
			since = time.time()
			for i in range(100):
				model(*infer_inputs)
			runtime = str(round((time.time()-since)*10, 2)) + ' ms'
	except Exception as e:
		print("On", n_devices, "devices")
		print("Inference Failed for:", model_name)
		if args.v:
			traceback.print_exc(file=sys.stdout)
	with open(model_name + str(n_devices) + '.txt','w+') as f:
		print(runtime, file=f)


if __name__ == "__main__":
	main()