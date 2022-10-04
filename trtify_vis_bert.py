
import torch
import trtorch
trtorch.logging.set_reportable_log_level(trtorch.logging.Level.Debug)
vb = VB()
input_shapes = 
model = vb.model.eval().cuda()
scripted_model = torch.jit.script(model)
compile_settings = {
"input_shapes": [input_shapes],
"op_precision": torch.float16
}
