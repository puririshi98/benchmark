from torchbenchmark import list_models
import torch
import trtorch
trtorch.logging.set_reportable_log_level(trtorch.logging.Level.Debug)
effnet=None
vt=None
for Model in list_models():
	if 'timm_efficientnet' in Model.name.lower():
		EF = Model
	if 'timm_vision_transformer' in Model.name.lower():
		VT = Model
ef = EF()
input_shapes = list(ef.cfg.infer_example_inputs.shape)
model = ef.model.eval().cuda()
scripted_model = torch.jit.script(model)

compile_settings = {
"input_shapes": [input_shapes],
"op_precision": torch.float16
}

trt_ts_module = trtorch.compile(scripted_model, compile_settings)
torch.jit.save(trt_ts_module, 'timm_efficientnet.jit')

vt = VT()
input_shapes = list(vt.cfg.infer_example_inputs.shape)
model = vt.model.eval().cuda()
scripted_model = torch.jit.script(model)

compile_settings = {
"input_shapes": [input_shapes],
"op_precision": torch.float16
}

trt_ts_module = trtorch.compile(scripted_model, compile_settings)
torch.jit.save(trt_ts_module, 'timm_vision_transformer.jit')
