import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
​
import math
​
torch._C._jit_set_nvfuser_enabled(True)
torch._C._jit_set_texpr_fuser_enabled(False)
torch._C._jit_set_profiling_executor(True)
torch._C._jit_set_profiling_mode(True)
torch._C._jit_override_can_fuse_on_cpu(False)
torch._C._jit_override_can_fuse_on_gpu(False)
torch._C._jit_set_bailout_depth(20)
​
class BertConfig :
    def __init__(self) :
        self.hidden_size = 1024
        self.num_attention_heads = 16
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.num_layers = 10
​
class Fusion(nn.Module):
    def __init__(self, config):
        super(Fusion, self).__init__()
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
​
    def forward(self, inputs, mask):
        out1 = inputs / math.sqrt(self.attention_head_size)
        out2 = out1 + mask
        out3 = F.softmax(out2, dim=-1)
        out4 = self.dropout(out3)
        return out4
​
# eager is 10 passes on data
# fuesd is 3
inner_dim = 197
if __name__ == "__main__" :
    inputs = torch.randn(1, 8, 197, inner_dim, device="cuda", dtype=torch.float, requires_grad=False)
    mask = torch.randn(1, 1, 1, inner_dim, device="cuda", dtype=torch.float, requires_grad=False)
    mask_bool = mask > 0.
   
    model = Fusion(BertConfig())
    model.cuda()
    model.eval()
   
    jit_model = torch.jit.script(model)
  
    for idx in range(10) :
        out = model(inputs, mask_bool)
​
    for idx in range(10) :
        out = jit_model(inputs, mask_bool)