import fairscale
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
import torch
import timm.models.vision_transformer
import timm.models.efficientnet
from transformers import BertModel, BertConfig
import argparse
from test_pipe import set_seed, gen_simple_linear_model, TimmConfigVT, TimmConfigEF
import traceback
import time
import os
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("MODEL_NAME", type=str,  help="model name")
	parser.add_argument("-v", action='store_true', default=False, help="Verbose")

	args = parser.parse_args()
	args.local_rank = int(os.environ["LOCAL_RANK"])
	torch.cuda.set_device(args.local_rank)
	device = torch.device("cuda", args.local_rank)
	torch.distributed.init_process_group(backend="nccl")
	n_devices = torch.distributed.get_world_size()
	set_seed()
	model_name = args.MODEL_NAME
	if model_name == 'EF':
		model = timm.create_model('mixnet_m', pretrained=False, scriptable=True)
		cfg = TimmConfigEF(model=model, precision='float32')
		infer_inputs = (cfg.infer_example_inputs.cuda(),)
	elif model_name == 'VT':
		model = timm.create_model('vit_small_patch16_224', pretrained=False, scriptable=True)
		cfg = TimmConfigVT(model=model, precision='float32')
		infer_inputs = (cfg.infer_example_inputs.cuda(),)		
	elif model_name == 'Linear':
		infer_inputs = (torch.randn((64,1024*8)).cuda(),)
		model = gen_simple_linear_model(n_devices)
	elif model_name == 'hugface':
		vocab_size = 30522
		batchsize = 64
		seqlen = 128
		infer_inputs = (torch.randint(low=0, high=vocab_size, size=(batchsize, seqlen)).long().cuda(),)
		model = BertModel(BertConfig())
	else:
		print("Model Not supported:", model_name)
	
	with open(model_name + str(n_devices) + '.txt','w+') as f:
		try:
			model = FSDP(model.cuda()).eval()
			with torch.cuda.amp.autocast():
				since = time.time()
				for i in range(100):
					model(*infer_inputs)
			runtime = str(round((time.time()-since)*10, 2)) + ' ms'
			print(runtime, file=f)
			print(runtime)
		except Exception as e:
			print("On", n_devices, "devices", file=f)
			print("Inference Failed for:", model_name, file=f)
			if args.v:
				traceback.print_exc(file=f)
				traceback.print_exc(file=sys.stdout)
	


if __name__ == "__main__":
	main()