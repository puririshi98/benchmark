import torch

dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=False)
torch.onnx.export(model, dummy_input, "res18.onnx", verbose=True)