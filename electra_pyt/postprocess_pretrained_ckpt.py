import argparse
import collections
import json
import os

import torch

from utils import log, heading
from run_pretrain import PretrainingConfig, PretrainingModel


def from_pretrained_ckpt(args):
    config = PretrainingConfig(
        model_name='none',
        data_dir='none',
        generator_hidden_size=0.3333333,
        vocab_size=30522
    )

    # Set up model
    model = PretrainingModel(config)

    # Load checkpoint
    try:
        checkpoint_suffix = args.pretrained_checkpoint.split("-")[-1].split("/")[0]
        step = int(checkpoint_suffix)
    except ValueError:
        step = "unknown"

    from collections import OrderedDict
    state_dict = OrderedDict()
    checkpoint = torch.load(os.path.join(args.pretrained_checkpoint, "checkpoint.pt"), map_location="cpu")
    for k, v in checkpoint["model"].items():
        name = k[7:] if k.startswith("module.") else k  # remove 'module.' of dataparallel
        state_dict[name] = v
    model.load_state_dict(state_dict)
    log(" ** Restored from {} at step {}".format(args.pretrained_checkpoint, step))

    disc_dir = os.path.join(args.output_dir, 'discriminator')
    gen_dir = os.path.join(args.output_dir, 'generator')

    heading(" ** Saving discriminator")
    # model.discriminator(model.discriminator.dummy_inputs)
    model.discriminator.save_pretrained(disc_dir)

    heading(" ** Saving generator")
    # model.generator(model.generator.dummy_inputs)
    model.generator.save_pretrained(gen_dir)


if __name__ == '__main__':
    # Parse essential args
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_checkpoint')
    parser.add_argument('--output_dir')
    args = parser.parse_args()

    from_pretrained_ckpt(args)