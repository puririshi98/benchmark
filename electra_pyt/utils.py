import json
import pickle
import sys
import os
import shutil
import unicodedata
import six
import time

import torch
import torch.distributed as dist


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def write_json(o, path):
    if "/" in path:
        mkdir(path.rsplit("/", 1)[0])
    with open(path, "w") as f:
        json.dump(o, f)


# def load_pickle(path):
#     with tf.io.gfile.GFile(path, "rb") as f:
#         return pickle.load(f)
#
#
# def write_pickle(o, path):
#     if "/" in path:
#         tf.io.gfile.makedirs(path.rsplit("/", 1)[0])
#     with tf.io.gfile.GFile(path, "wb") as f:
#         pickle.dump(o, f, -1)

def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_main_process():
    return get_rank() == 0


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def format_step(step):
    if isinstance(step, str):
        return step
    s = ""
    if len(step) > 0:
        s += "Training Epoch: {} ".format(step[0])
    if len(step) > 1:
        s += "Training Iteration: {} ".format(step[1])
    if len(step) > 2:
        s += "Validation Iteration: {} ".format(step[2])
    return s


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def rmrf(path):
    if os.path.exists(path):
        shutil.rmtree(path)


def rmkdir(path):
    rmrf(path)
    mkdir(path)


def log(*args, **kwargs):
    all_rank = kwargs.pop("all_rank", False)
    if not all_rank and not is_main_process():
        return
    msg = " ".join(map(str, args))
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def log_config(config):
    for key, value in sorted(config.__dict__.items()):
        log(key, value)
    log()


def heading(*args):
    log(80 * "=")
    log(*args)
    log(80 * "=")


def nest_dict(d, prefixes, delim="_"):
    """Go from {prefix_key: value} to {prefix: {key: value}}."""
    nested = {}
    for k, v in d.items():
        for prefix in prefixes:
            if k.startswith(prefix + delim):
                if prefix not in nested:
                    nested[prefix] = {}
                nested[prefix][k.split(delim, 1)[1]] = v
            else:
                nested[k] = v
    return nested


def flatten_dict(d, delim="_"):
    """Go from {prefix: {key: value}} to {prefix_key: value}."""
    flattened = {}
    for k, v in d.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                flattened[k + delim + k2] = v2
        else:
            flattened[k] = v
    return flattened


def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, unicode):
            return text.encode("utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def get_readable_time(elapsed):
    d, h, m, s = [int(x) for x in time.strftime("%d:%H:%M:%S", time.gmtime(elapsed)).split(':')]
    d -= 1
    return '{:2d}h{:2d}m{:2d}s'.format(24*d + h, m, s)
