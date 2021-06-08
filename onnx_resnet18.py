import torch
import torchvision.models as models

dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
model = models.resnet18()
torch.onnx.export(model, dummy_input, "res18.onnx", verbose=True)