import torch
import sys
import torchvision.models as models
import torchvision
import time
print("Loading Model...")
model = torch.nn.Sequential(torch.nn.Linear(128,64),torch.nn.ReLU(),torch.nn.Linear(64,10))
print("Loading Dataset...")
data=torch.randn((1000,128)).cuda()
labels = torch.randint((10),(1000,)).long()
# imagenet_data = torchvision.datasets.ImageNet(sys.argv[1])
# data_loader = torch.utils.data.DataLoader(imagenet_data,
#                                           batch_size=32,
#                                           shuffle=True,
#                                           num_workers=4)
model = model.train().cuda()
print("FP32 convergence:")
optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
loss=torch.nn.CrossEntropyLoss()
torch.manual_seed(0)
time_sum=0
for epoch in range(600):
	since=time.time()
	optimizer.zero_grad()
	indys=torch.randint(len(data), (32,))
	batch=data[indys].cuda()
	label = labels[indys].cuda()
	output = model(batch)
	if isinstance(output, tuple):
		output = output[0]
	l=loss(output, label)
	l.backward()
	optimizer.step()
	time_sum+=time.time()-since
	if epoch%200==0:
		print(l)
		print("Time Per Iter:",round(1000.0*time_sum/(epoch+1),2),"ms")
model = torch.nn.Sequential(torch.nn.Linear(128,64),torch.nn.ReLU(),torch.nn.Linear(64,10))
model = model.train().cuda().bfloat16()
print("Bfloat16 convergence:")
optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
loss=torch.nn.CrossEntropyLoss()
torch.manual_seed(0)
time_sum=0
for epoch in range(600):
	since=time.time()
	optimizer.zero_grad()
	indys=torch.randint(len(data), (32,))
	batch=data[indys].cuda().bfloat16()
	label = labels[indys].cuda()
	output = model(batch)
	if isinstance(output, tuple):
		output = output[0]
	l=loss(output, label)
	l.backward()
	optimizer.step()
	time_sum+=time.time()-since
	if epoch%200==0:
		print(l)
		print("Time Per Iter:",round(1000.0*time_sum/(epoch+1),2),"ms")
model = torch.nn.Sequential(torch.nn.Linear(128,64),torch.nn.ReLU(),torch.nn.Linear(64,10))
model = model.train().cuda()
print("AMP Bfloat16 convergence:")
optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
loss=torch.nn.CrossEntropyLoss()
torch.manual_seed(0)
scaler = torch.cuda.amp.GradScaler(enabled=True)
time_sum=0
for epoch in range(600):
	since=time.time()
	optimizer.zero_grad()
	indys=torch.randint(len(data), (32,))
	batch=data[indys].cuda()
	with torch.cuda.amp.autocast(enabled=True, bfloat=True):
		label = labels[indys].cuda()
		output = model(batch)
		if isinstance(output, tuple):
			output = output[0]
		l=loss(output, label)

		scaler.scale(l).backward()
		scaler.step(optimizer)
		optimizer.zero_grad()
		scaler.update()
		time_sum+=time.time()-since
		if epoch%200==0:
			print(l)
			print("Time Per Iter:",round(1000.0*time_sum/(epoch+1),2),"ms")

model = torch.nn.Sequential(torch.nn.Linear(128,64),torch.nn.ReLU(),torch.nn.Linear(64,10))
model = model.train().cuda()
print("AMP fp16 convergence:")
optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
loss=torch.nn.CrossEntropyLoss()
torch.manual_seed(0)
scaler = torch.cuda.amp.GradScaler(enabled=True)
time_sum=0
for epoch in range(600):
	since=time.time()
	optimizer.zero_grad()
	indys=torch.randint(len(data), (32,))
	batch=data[indys].cuda()
	with torch.cuda.amp.autocast(enabled=True):
		label = labels[indys].cuda()
		output = model(batch)
		if isinstance(output, tuple):
			output = output[0]
		l=loss(output, label)

		scaler.scale(l).backward()
		scaler.step(optimizer)
		optimizer.zero_grad()
		scaler.update()
		time_sum+=time.time()-since
		if epoch%200==0:
			print(l)
			print("Time Per Iter:",round(1000.0*time_sum/(epoch+1),2),"ms")

model = torch.nn.Sequential(torch.nn.Linear(128,64),torch.nn.ReLU(),torch.nn.Linear(64,10))
model = model.train().cuda().half()
print("FP16 convergence:")
optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
loss=torch.nn.CrossEntropyLoss()
torch.manual_seed(0)
time_sum=0

for epoch in range(600):
	since=time.time()
	optimizer.zero_grad()
	indys=torch.randint(len(data), (32,))
	batch=data[indys].cuda().half()
	label = labels[indys].cuda()
	output = model(batch)
	if isinstance(output, tuple):
		output = output[0]
	l=loss(output, label)
	
	l.backward()
	optimizer.step()
	time_sum+=time.time()-since
	if epoch%200==0:
		print(l)
		print("Time Per Iter:",round(1000.0*time_sum/(epoch+1),2),"ms")