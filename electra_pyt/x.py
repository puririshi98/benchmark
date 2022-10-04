import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import time
from statistics import *

# model_name->str: torchvision name string for a model like 'alexnet'
# shape->tuple(int, int, int): A 3 number tuple (channels, height, width)
# classes->int: number of classes for labels
def preprocess(model):
    torch.cuda.init()
    model = model.cuda()
    model.train()
    return model

def imagenet(shape, is_training=False, **kwargs):
    classes = kwargs.get('classes', None)
    if len(shape) != 3 :
        raise ValueError('Shape Tuple is not of size 3.')
    i=0
    while i < 500:
        i+=1
        yield torch.randn((args.batchsize, shape[0], shape[1], shape[2])).cuda()


def run_model(model_name, shape, classes, benchmark=True):
    torch.backends.cudnn.benchmark = benchmark
    data = imagenet(shape, is_training=True, classes=classes)

    model_to_call = getattr(models, model_name)
    # Don't need to use pretrained wieghts, starting from scratch is good
    model = None
    model = model_to_call(pretrained=False)
    model = preprocess(model)

    criterion  = nn.CrossEntropyLoss()
    optimizer  = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    times = []
    for inputs, labels in data:
        since=time.time()
        optimizer.zero_grad()

        outputs = model(inputs)
        if not isinstance(outputs, torch.Tensor) and hasattr(outputs, 'logits'):
            outputs = outputs.logits
        loss = criterion(outputs, labels)

        loss.backward()
            # TODO:  The line below works functionally, but I'm worried it might
            # take too much time on the runners.  Restore when we confirm
            # the runners can handle it.
            # time_bwd_ops(loss).backward()

        optimizer.step()
        times.append(time.time()-since)
        if len(times)>35:
            break
    
    times=times[5:] #Ignore first 5 iters as 'warmups'
    print("Mean, StDev, and Max of of Iteration Times:", [mean(times), stdev(times), max(times)])

models299 = ['googlenet','inception_v3']

models224 = ['alexnet','densenet161','mnasnet1_0','mobilenet_v2','resnet18','resnext50_32x4d','shufflenet_v2_x1_0','squeezenet1_0','vgg16','wide_resnet50_2']
shapes = [(3, 224, 224)]*len(models224) + [(3, 299, 299)] * len(models299)
model_list = models224+models299
for model, shape in zip(model_list,shapes):
    run_model(model, shape, 1000)

