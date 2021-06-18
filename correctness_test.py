import torch


print("Conv2D Correctness Test:")
input=torch.randn((1,3,224,224)).cuda()
weight=torch.randn((5,3,5,5)).cuda()
bf=torch.nn.functional.conv2d(input.bfloat16(), weight.bfloat16())
fp32=torch.nn.functional.conv2d(input.float(), weight.float())
fp16=torch.nn.functional.conv2d(input.half(), weight.half())

print("mean(FP32-BF16): ",abs(float(torch.mean(fp32-bf).cpu())))
print("mean(FP32-FP16): ",abs(float(torch.mean(fp32-fp16).cpu())))
print("Linear layer correctness test:")
input=torch.randn((1,224)).cuda()
weight=torch.randn((12,224)).cuda()
bf=torch.nn.functional.linear(input.bfloat16(), weight.bfloat16())
fp32=torch.nn.functional.linear(input.float(), weight.float())
fp16=torch.nn.functional.linear(input.half(), weight.half())
print("mean(FP32-BF16): ",abs(float(torch.mean(fp32-bf).cpu())))
print("mean(FP32-FP16): ",abs(float(torch.mean(fp32-fp16).cpu())))

print("softmax correctness test")

input=torch.randn((1,224)).cuda()
bf=torch.nn.functional.softmax(input.bfloat16())
fp32=torch.nn.functional.softmax(input.float())
fp16=torch.nn.functional.softmax(input.half())

print("mean(FP32-BF16): ",abs(float(torch.mean(fp32-bf).cpu())))
print("mean(FP32-FP16): ",abs(float(torch.mean(fp32-fp16).cpu())))