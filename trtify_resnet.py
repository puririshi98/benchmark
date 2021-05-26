import torch
import trtorch
precision = 'fp16'
model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
input_shapes = [1, 3, 224, 224]
model = model.eval().cuda()
scripted_model = torch.jit.script(model)

compile_settings = {
"input_shapes": [input_shapes],
"op_precision": torch.float16
}

trt_ts_module = trtorch.compile(scripted_model, compile_settings)
torch.jit.save(trt_ts_module, 'resnet.jit')