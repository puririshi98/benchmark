import torch
import sys
import torchvision.models as models
import torchvision
print("Loading Model...")
model = models.resnet18().cuda()
print("Loading Dataset...")
data=torch.randn((1000,3,224,224)).cuda()
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
for epoch in range(100000):
	optimizer.zero_grad()
	indys=torch.randint(len(data), (32,))
	batch=data[indys].cuda()
	label = labels[indys].cuda()
	output = model(batch)
	if isinstance(output, tuple):
		output = output[0]
	l=loss(output, label)
	if epoch%25000==0:
		print(l)
	l.backward()
	optimizer.step()
model = models.resnet18().cuda()
model = model.train().cuda().bfloat16()
print("Bfloat16 convergence:")
optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
loss=torch.nn.CrossEntropyLoss()
torch.manual_seed(0)
for epoch in range(100000):
	optimizer.zero_grad()
	indys=torch.randint(len(data), (32,))
	batch=data[indys].cuda().bfloat16()
	label = labels[indys].cuda()
	output = model(batch)
	if isinstance(output, tuple):
		output = output[0]
	l=loss(output, label)
	if epoch%25000==0:
		print(l)
	l.backward()
	optimizer.step()
model = models.resnet18().cuda()
model = model.train().cuda().half()
print("FP16 convergence:")
optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
loss=torch.nn.CrossEntropyLoss()
torch.manual_seed(0)
for epoch in range(100000):
	optimizer.zero_grad()
	indys=torch.randint(len(data), (32,))
	batch=data[indys].cuda().half()
	label = labels[indys].cuda()
	output = model(batch)
	if isinstance(output, tuple):
		output = output[0]
	l=loss(output, label)
	if epoch%25000==0:
		print(l)
	l.backward()
	optimizer.step()



