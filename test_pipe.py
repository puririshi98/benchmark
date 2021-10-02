import torch
import timm.models.vision_transformer
import timm.models.efficientnet
from transformers import BertModel, BertConfig

# Generated by gen_timm_models.py
import torch.nn as nn
from torch.nn import *
import dataclasses
import time
import torch.distributed.pipeline
import torch.distributed.pipeline.sync
import os

def resolve_precision(precision: str):
    assert precision in ('amp', 'float16', 'bfloat16', 'float32')
    use_amp = False
    model_dtype = torch.float32
    data_dtype = torch.float32
    if precision == 'amp':
        use_amp = True
    elif precision == 'float16':
        model_dtype = torch.float16
        data_dtype = torch.float16
    elif precision == 'bfloat16':
        model_dtype = torch.bfloat16
        data_dtype = torch.bfloat16
    return use_amp, model_dtype, data_dtype

@dataclasses.dataclass
class OptimizerOption:
    lr: float
    opt: str
    weight_decay: float
    momentum: float

class TimmConfigEF:
    def _init_input(self):
        self.example_inputs = torch.randn(
            (self.batch_size,) + self.input_size, dtype=self.data_dtype)
        self.infer_example_inputs = torch.randn(
            (self.batch_size,) + self.input_size, dtype=self.data_dtype)

    def __init__(self, model, precision):
        self.model = model
        self.use_amp, self.model_dtype, self.data_dtype = resolve_precision(precision)
        # Configurations
        self.batch_size = 64
        self.num_classes = self.model.num_classes
        self.target_shape = tuple()
        self.input_size = self.model.default_cfg["input_size"]
        self._init_input()

class TimmConfigVT:
    def _init_input(self):
        self.example_inputs = torch.randn(
            (self.batch_size,) + self.input_size, dtype=self.data_dtype)
        self.infer_example_inputs = torch.randn(
            (self.batch_size,) + self.input_size, dtype=self.data_dtype)

    def __init__(self, model, precision):
        self.model = model
        self.use_amp, self.model_dtype, self.data_dtype = resolve_precision(precision)
        # Configurations
        self.batch_size = 64
        self.num_classes = self.model.num_classes
        self.target_shape = tuple()
        self.input_size = self.model.default_cfg["input_size"]
        self._init_input()

def gen_simple_linear_model(n_devices):
	layer_list = []
	hsize = 1024*8
	for i in range(n_devices):
		layer_list += [torch.nn.Linear(hsize,int(hsize))]
		if i != n_devices-1:
			layer_list += [torch.nn.ReLU()]
	return torch.nn.Sequential(*layer_list)

def not_custom_block(module):
	for typeofmodule in dir(torch.nn):
		if isinstance(module,eval(typeofmodule)):
			return True
	return False

def assign_chunks(modules, n_devices):
	num_modules = len(modules)
	new_Module = torch.nn.Sequential(*modules)
	modules = [module for module in model.modules() if ((not isinstance(module, nn.Sequential)) and not_custom_block(module))]

	modules_in_each_chunk = int(num_modules / n_devices)
	start_ptr = 0
	for chunk in range(n_devices):
		if chunk == n_devices - 1:
			chunks = modules[start_ptr:]
		else:
			chunks = modules[start_ptr:(start_ptr+modules_in_each_chunk)]
		for module_x in chunks:
			module_x.cuda(chunk)
		start_ptr += modules_in_each_chunk
	return new_Module



def main():
	os.environ['MASTER_ADDR'] = 'localhost'
	os.environ['MASTER_PORT'] = '29500'
	torch.distributed.rpc.init_rpc('worker', rank=0, world_size=1)
	runtimes = {'EF':{}, 'VT':{}, 'Linear':{}, 'hugface':{}}
	for n_devices in range(1,torch.cuda.device_count()+1):
		print("Testing", n_devices,"devices:")
		#Model Inits
		models = {'EF':timm.create_model('mixnet_m', pretrained=False, scriptable=True),
			'VT':timm.create_model('vit_small_patch16_224', pretrained=False, scriptable=True),
			'Linear':gen_simple_linear_model(n_devices),
			'hugface': BertModel(BertConfig())}


		for model_name, model in models.items():
			#Model Setup
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
				infer_inputs = (
            		torch.randint(0, vocab_size, (batchsize, seqlen)).cuda(),
           			torch.randint(0, 2, (batchsize, seqlen)).cuda(),
            		torch.randint(0, 2, (batchsize, seqlen)).cuda()
        		)
			else:
				print("Model Not supported:", model_name)

			#Chunk if distributing across gpus
			if n_devices > 1:
				try:
					modules = [module for module in model.modules() if ((not isinstance(module, nn.Sequential)) and not_custom_block(module))]
					model = assign_chunks(modules, n_devices)	
					model = torch.distributed.pipeline.sync.Pipe(model, chunks=n_devices, checkpoint='except_last', deferred_batch_norm=False)
				except Exception as e:
					print("On", n_devices, "devices")
					print("Could Not Succesfully Breakup:", model_name)
					print(e)
			else:
				model =  model.cuda()
				print("-*"*25)
				print(model_name)
				print([module for module in model.modules() if ((not isinstance(module, nn.Sequential)) and not_custom_block(module))])
				print("-*"*25)
			model = model.eval()

			#time it
			try:
				since = time.time()
				for i in range(100):
					model(*infer_inputs)
				runtimes[model_name][str(n_devices) + '_gpus'] = str(round((time.time()-since)*10, 2)) + ' ms'
			except Exception as e:
				print("On", n_devices, "devices")
				print("Inference Failed for:", model_name)
				print(e)
		print()
		print('#'*25)
	#report it
	print(runtimes)




if __name__ == "__main__":
	main()