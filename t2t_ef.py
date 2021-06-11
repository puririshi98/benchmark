import torch
from torch2trt import torch2trt
import timm.models.efficientnet
import timm
import time
import numpy as np
# create some regular pytorch model...
model = timm.create_model('mixnet_m', pretrained=False, scriptable=True).float().eval().cuda()

# create example data
x = torch.ones((1, 3, 224, 224)).cuda()

# convert to TensorRT feeding sample data as input
model_trt = torch2trt(model, [x])
timesum=0
for i in range(5):
	since=time.time()
	y = model(x)
	timesum+=time.time()-since
print("Vanila torch fp32 avg time:",round(1000.0*timesum/5.0,4),"ms")
timesum=0
for i in range(5):
	since=time.time()
	y_trt = model_trt(x)
	timesum+=time.time()-since
print("torch2trt fp32 avg time:",round(1000.0*timesum/5.0,4),"ms")

time_sum=0
torch.backends.cudnn.benchmark = True
model1=model.float().eval()
s = torch.cuda.Stream()
torch.cuda.synchronize()
inputs = torch.tensor(np.random.random((1, 3, input_size, input_size))).cuda().float()
with torch.cuda.stream(s):
	for _ in range(5):
		out=model1(inputs)
	torch.cuda.empty_cache()
	g = torch.cuda._Graph()
	torch.cuda.synchronize()
	g.capture_begin()
	out=model1(inputs)
	g.capture_end()
	torch.cuda.synchronize()

for _ in range(5):
	t1=time.time()
	g.replay()
	torch.cuda.synchronize()
	
	time_sum+=time.time()-t1
print("using cudagraphsfp32 mode:")
print("avg cost time: ", round(1000.0*time_sum/5.0,4),'ms')

