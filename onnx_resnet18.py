import torch

dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True)