import torch
import time

def BS12_sorted():
    x = torch.randn((60,201600), dtype=torch.float16, device="cuda", requires_grad=True)
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(1000):
        y,z = x.topk(k=2000, dim=1, sorted=True)
    torch.cuda.synchronize()
    time_for_topk = (time.time() - start_time) / 1000
    print(time_for_topk)

def BS12_unsorted():
    x = torch.randn((120000), dtype=torch.float32, device="cuda", requires_grad=True)
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(1000):
        y,z = x.topk(k=12000, dim=0, sorted=False)
    torch.cuda.synchronize()
    time_for_topk = (time.time() - start_time) / 1000
    print(time_for_topk)

def BS1_sorted():
    x = torch.randn((5,201600), dtype=torch.float16, device="cuda", requires_grad=True)
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(1000):
        y,z = x.topk(k=2000, dim=1, sorted=True)
    torch.cuda.synchronize()
    time_for_topk = (time.time() - start_time) / 1000
    print(time_for_topk)

def BS1_unsorted():
    x = torch.randn((1, 10000), dtype=torch.float32, device="cuda", requires_grad=True)
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(1000):
        y,z = x.topk(k=1000, dim=1, sorted=False)
    torch.cuda.synchronize()
    time_for_topk = (time.time() - start_time) / 1000
    print(time_for_topk)

BS12_sorted()
BS12_unsorted()
BS1_sorted()
BS1_unsorted()