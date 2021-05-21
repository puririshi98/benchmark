import torch
import trtorch
precision = 'fp16'
ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math=precision)
input_shapes = [3, 300, 300]
model = ssd_model.eval().cuda()
scripted_model = torch.jit.script(model)

compile_settings = {
"input_shapes": [input_shapes],
"op_precision": torch.float16
}

trt_ts_module = trtorch.compile(scripted_model, compile_settings)
torch.jit.save(trt_ts_module, 'ssd.jit')