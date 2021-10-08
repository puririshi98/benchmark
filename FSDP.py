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



if __name__ == "__main__":
	main()