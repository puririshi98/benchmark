"""
A lightweight runner that just sets up a model and runs one of its functions in a particular configuration.

Intented for debugging/exploration/profiling use cases, where the test/measurement harness is overhead.

DANGER: make sure to `python install.py` first or otherwise make sure the benchmark you are going to run
        has been installed.  This script intentionally does not automate or enforce setup steps.

Wall time provided for sanity but is not a sane benchmark measurement.
"""
import argparse
import time
import torch.autograd.profiler as profiler
from conftest import set_fuser
from torchbenchmark import list_models
import torch

def run_one_step(func, precision='fp16', graphs=False, bench=False):
    t0 = time.time()
    func(precision=precision, graphs=graphs, bench=bench)
    t1 = time.time()
    print(f"Ran in {t1 - t0} seconds.")

def profile_one_step(func, nwarmup=3):
    for i in range(nwarmup):
        func()

    with profiler.profile(record_shapes=True) as prof:
        func()

    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("model", help="Full or partial name of a model to run.  If partial, picks the first match.")
    parser.add_argument("-d", "--device", choices=["cpu",  "cuda"], default="cpu", help="Which device to use.")
    parser.add_argument("-m", "--mode", choices=["eager",  "jit"], default="eager", help="Which mode to run.")
    parser.add_argument("-t", "--test", choices=["eval",  "train"], default="eval", help="Which test to run.")
    parser.add_argument("-fuser", choices=["te",  "nv"], default="nv", help="Which fuser to run.")
    parser.add_argument("-graphs", action="store_true", help="use cudagraphs")
    parser.add_argument("-precision", choices=["fp32",  "fp16", "bfloat16"], default="fp16", help="Which precision to run in (fp16 vs fp32).")
    parser.add_argument("-cudnnbenchmark", action="store_true",  help="turn cudnn benchmark on")
    parser.add_argument("--profile", action="store_true", help="Run the profiler around the function")
    parser.add_argument("-batchsize", default=1, type=int,help="Increase batchsize from the default of 1, (only works for visiontransformer rn)")
    parser.add_argument("-large", action="store_true", help="to use with BERT to switch to BERT large from base")
    parser.add_argument("-seqlen", default=128, type=int, help="seqlen for Language models")
    args = parser.parse_args()
    print(args)
    torch.cuda.cudart().cudaProfilerStart()

    found = False
    for Model in list_models():
        if args.model.lower() in Model.name.lower():
            found = True
            break
    if found:
        print(f"Running {args.test} method from {Model.name} on {args.device} in {args.mode} mode")
    else:
        print(f"Unable to find model matching {args.model}")
        exit(-1)
    if args.mode == 'jit':
        set_fuser(args.fuser)
    # build the model and get the chosen test method
    if args.model.lower() == 'hf_bert':
        m = Model(args.device, jit=(args.mode=="jit"), batchsize=int(args.batchsize), seqlen=args.seqlen, large=args.large)
    else:
        if args.batchsize!=1:
            m = Model(args.device, jit=(args.mode=="jit"), batchsize=int(args.batchsize))
        else:
            m = Model(args.device, jit=(args.mode=="jit"))
    test = getattr(m, args.test)

    run_one_step(test, precision=args.precision, graphs=args.graphs, bench=args.cudnnbenchmark)
    torch.cuda.cudart().cudaProfilerStop()



