import torch
import trtorch
precision = 'fp16'
model = torch.nn.Sequential(torch.nn.Linear(1024,128),torch.nn.ReLU(), torch.nn.Linear(128,1))
input_shapes = [1, 1024]
model = model.eval().cuda()
scripted_model = torch.jit.script(model)

compile_settings = {
"input_shapes": [input_shapes],
"op_precision": torch.float16
}

trt_ts_module = trtorch.compile(scripted_model, compile_settings)
torch.jit.save(trt_ts_module, 'mlp.jit')