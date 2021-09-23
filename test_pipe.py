import torch
import timm.models.vision_transformer
import timm.models.efficientnet
from transformers import BertModel, BertConfig

# Generated by gen_timm_models.py
import torch.nn as nn
import dataclasses
import time
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
            (1,) + self.input_size, dtype=self.data_dtype)

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
            (1,) + self.input_size, dtype=self.data_dtype)

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
	layer_list = [torch.nn.Linear(1024,512), torch.nn.ReLU()]
	hsize=512
	for i in range(n_devices):

		layer_list += [torch.nn.Linear(hsize,int(hsize/2))]
		if i != n_devices-1:
			layer_list += [torch.nn.ReLU()]
		hsize = int(hsize/2)
	return torch.nn.Sequential(*layer_list)

def assign_chunks(modules, n_devices):
	num_modules = len(modules)
	modules_in_each_chunk = int(num_modules / n_devices)
	start_ptr = 0
	for chunk in range(n_devices):
		if chunk == n_devices - 1:
			chunks = modules[start_ptr:]
		else:
			chunks = modules[start_ptr:(start_ptr+modules_in_each_chunk)]
		for module_x in chunk:
			module_x.cuda(chunk)
		start_ptr += modules_in_each_chunk



def main():
	runtimes = {'EF':{}, 'VT':{}, 'Linear':{}, 'hugface':{}}
	for n_devices in range(1,torch.cuda.device_count()):
		#Model Inits
		models = {'EF':timm.create_model('mixnet_m', pretrained=False, scriptable=True),
			'VT':timm.create_model('vit_small_patch16_224', pretrained=False, scriptable=True),
			'Linear':gen_simple_linear_model(n_devices),
			'hugface': BertModel(BertConfig())}


		for model_name, model in models.items():
			#Model Setup
			if model_name == 'EF':
				cfg = TimmConfigEF(model=model, precision='float32')
				infer_inputs = (cfg.infer_example_inputs.cuda())
			elif model_name == 'VT':
				cfg = TimmConfigVT(model=model, precision='float32')
				infer_inputs = (cfg.infer_example_inputs.cuda())
			elif model_name == 'Linear':
				infer_inputs = (torch.randn((64,1024)).cuda())
			elif model_name == 'hugface':
				infer_inputs = (
            		torch.randint(0, config.vocab_size, (batchsize, seqlen)).cuda(),
           			torch.randint(0, 2, (batchsize, seqlen)).cuda(),
            		torch.randint(0, 2, (batchsize, seqlen)).cuda()
        		)
			else:
				print("Model Not supported:", model_name)

			#Chunk if distributing across gpus
			try:
				if n_devices > 1:
					modules = [module for module in model.modules() if not isinstance(module, nn.Sequential)]
					assign_chunks(modules)	
					model = torch.distributed.pipeline.sync.Pipe(model, chunks=n_devices, checkpoint='except_last', deferred_batch_norm=False)
			except Exception as e:
				print("On", n_devices, "devices")
				print("Could Not Succesfully Breakup:", model_name)
				print(e)

			#time it
			try:
				since = time.time()
				for i in range(100):
					model(*infer_inputs)
				runtimes[model_name][str(n_devices) + '_gpus'] = round((since - time.time())/100.0, 2)
			except:
				print("On", n_devices, "devices")
				print("Inference Failed for:", model_name)
				print(e)
	#report it
	print(runtimes)




if __name__ == "__main__":
	main()