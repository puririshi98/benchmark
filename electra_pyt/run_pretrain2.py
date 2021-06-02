# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pre-trains an ELECTRA model."""

import time
import datetime
import argparse
import logging
import os
import glob
import random
import collections
import itertools

import torch
import numpy as np
from torch import Tensor, device, dtype, nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
# from apex import amp
from apex.optimizers import FusedAdam, FusedLAMB

import pretrain_utils
import utils
from utils import log, get_rank, get_world_size, is_main_process, barrier
from configuration import ElectraConfig
from optimization import AdamW, get_linear_schedule_with_warmup, get_poly_schedule_with_warmup
from modeling import ElectraForPreTraining, ElectraForMaskedLM
import torch.cuda.profiler as profiler
from dllogger import JSONStreamBackend
import dllogger
import json

logger = logging.getLogger(__name__)

torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)

scaler = torch.cuda.amp.GradScaler()

def setup_logger(args):
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, 'dllogger_rank{}.log'.format(get_rank()))

    dllogger.init(backends = [JSONStreamBackend(verbosity=1, filename=log_path)])
    for k,v in vars(args).items():
        dllogger.log(step='PARAMETER', data={k:v}, verbosity=0)

    container_setup_info = {
        'NVIDIA_PYTORCH_VERSION': os.environ.get('NVIDIA_PYTORCH_VERSION'),
        'PYTORCH_VERSION': os.environ.get('PYTORCH_VERSION'),
        'CUBLAS_VERSION': os.environ.get('CUBLAS_VERSION'),
        'NCCL_VERSION': os.environ.get('NCCL_VERSION'),
        'CUDA_DRIVER_VERSION': os.environ.get('CUDA_DRIVER_VERSION'),
        'CUDNN_VERSION': os.environ.get('CUDNN_VERSION'),
        'CUDA_VERSION': os.environ.get('CUDA_VERSION'),
        'NVIDIA_PIPELINE_ID': os.environ.get('NVIDIA_PIPELINE_ID'),
        'NVIDIA_BUILD_ID': os.environ.get('NVIDIA_BUILD_ID'),
        'NVIDIA_TF32_OVERRIDE': os.environ.get('NVIDIA_TF32_OVERRIDE'),
    }
    dllogger.log(step='PARAMETER', data=container_setup_info, verbosity=0)

def postprocess_dllog(args):
    logfiles = [open(os.path.join(args.log_dir, 'dllogger_rank{}.log'.format(i)), 'r') for i in range(get_world_size())]
    with open(os.path.join(args.log_dir, 'dllogger.log'), 'w') as dest_file:
        for lines in zip(*[f.readlines() for f in logfiles]):
            json_lines = [json.loads(l[5:]) for l in lines]
            assert all(x['step'] == json_lines[0]['step'] for x in json_lines)
            if json_lines[0]['step'] == 'PARAMETER':
                dest_file.write(lines[0])
            else:
                d =  dict.fromkeys(json_lines[0]['data'])
                for k in d.keys():
                    vs = [line['data'][k] for line in json_lines]
                    d[k] = sum(vs)/len(vs)
                json_lines[0]['data'] = d
                dest_file.write('DLLL ')
                dest_file.write(json.dumps(json_lines[0]))
                dest_file.write('\n')

    for l in logfiles:
        l.close()


class PretrainingConfig(object):
    """Defines pre-training hyperparameters."""

    def __init__(self, model_name, data_dir, **kwargs):
        self.model_name = model_name
        self.data_dir = data_dir
        self.results_dir = 'results'
        self.seed = 42

        self.debug = False  # debug mode for quickly running things
        self.do_train = True  # pre-train ELECTRA
        self.do_eval = False  # evaluate generator/discriminator on unlabeled data
        self.phase2 = False

        # device
        self.local_rank = -1
        self.n_gpu = 1
        self.no_cuda = True

        # amp
        self.amp = True
        self.amp_opt_level = "O1"

        # optimizer type
        self.optimizer = "adam"

        # lamb whitelisting for LN and biases
        self.skip_adaptive = False

        # loss functions
        self.electra_objective = True  # if False, use the BERT objective instead
        self.gen_weight = 1.0  # masked language modeling / generator loss
        self.disc_weight = 50.0  # discriminator loss
        self.mask_prob = 0.15  # percent of input tokens to mask out / replace

        # optimization
        self.learning_rate = 2e-4
        self.lr_decay_power = 1.0  # linear weight decay by default
        self.weight_decay_rate = 0.01
        self.num_warmup_steps = 10000
        self.opt_beta_1 = 0.9
        self.opt_beta_2 = 0.999
        self.end_lr = 0.0

        # training settings
        self.log_freq = 100
        self.save_checkpoints_steps = 1000
        self.num_train_steps = 1000000
        self.gradient_accumulation_steps = 1
        self.keep_checkpoint_max = 5  # maximum number of recent checkpoint files to keep;  change to 0 or None to keep all checkpoints
        self.restore_checkpoint = None
        # self.load_weights = False

        # model settings
        self.model_size = "small"  # one of "small", "base", or "large"
        # override the default transformer hparams for the provided model size; see
        # modeling.BertConfig for the possible hparams and util.training_utils for
        # the defaults
        self.model_hparam_overrides = (
            kwargs["model_hparam_overrides"]
            if "model_hparam_overrides" in kwargs else {})
        self.vocab_size = 30522  # number of tokens in the vocabulary
        self.do_lower_case = True  # lowercase the input?

        # generator settings
        self.uniform_generator = False  # generator is uniform at random
        self.shared_embeddings = True  # share generator/discriminator token embeddings?
        # self.untied_generator = True  # tie all generator/discriminator weights?
        self.generator_layers = 1.0  # frac of discriminator layers for generator
        self.generator_hidden_size = 0.25  # frac of discrim hidden size for gen
        self.disallow_correct = False  # force the generator to sample incorrect
        # tokens (so 15% of tokens are always
        # fake)
        self.temperature = 1.0  # temperature for sampling from generator

        # batch sizes
        self.max_seq_length = 128
        self.train_batch_size = 128
        self.eval_batch_size = 128

        self.update(kwargs)
        # default locations of data files
        self.pretrain_hdf5 = os.path.join(
            data_dir, "pretrain_hdf5/*.hdf5")
        self.vocab_file = os.path.join(data_dir, "vocab.txt")
        self.model_dir = os.path.join(self.results_dir, model_name)
        self.checkpoints_dir = os.path.join(self.model_dir, "checkpoints")
        self.log_dir = os.path.join(self.model_dir, "logs")

        self.max_predictions_per_seq = int((self.mask_prob + 0.005) *
                                           self.max_seq_length)

        # passed-in-arguments override defaults
        self.update(kwargs)

    def update(self, kwargs):
        for k, v in kwargs.items():
            if v is not None:
                self.__dict__[k] = v


class PretrainingModel(nn.Module):
    """Transformer pre-training using the replaced-token-detection task."""

    def __init__(self, config):
        super().__init__()
        # Set up model config
        self._config = config
        self.disc_config = ElectraConfig(vocab_size=config.vocab_size,
                                         embedding_size=768,
                                         hidden_size=768,
                                         num_hidden_layers=12,
                                         num_attention_heads=12,
                                         intermediate_size=3072,
                                         hidden_act="gelu",
                                         hidden_dropout_prob=0.1,
                                         attention_probs_dropout_prob=0.1, )

        # Set up discriminator
        self.discriminator = ElectraForPreTraining(self.disc_config)

        # Set up generator
        gen_config = get_generator_config(config, self.disc_config)
        if config.electra_objective:
            if config.shared_embeddings:
                self.generator = ElectraForMaskedLM(
                    gen_config, shared_embeddings=True,
                    input_embeddings=self.discriminator.get_embeddings())
            else:
                self.generator = ElectraForMaskedLM(gen_config)
        else:
            self.generator = ElectraForMaskedLM(self.disc_config)

    def forward(self, features):
        config = self._config

        # Mask the input
        masked_inputs = pretrain_utils.mask(
            config, pretrain_utils.features_to_inputs(features), config.mask_prob)
        # Generator
        if config.uniform_generator:
            mlm_output = self._get_masked_lm_output(masked_inputs, None)
        else:
            mlm_output = self._get_masked_lm_output(
                masked_inputs, self.generator)
        fake_data = self._get_fake_data(masked_inputs, mlm_output.logits)
        total_loss = config.gen_weight * mlm_output.loss

        # Discriminator
        disc_output = None
        if config.electra_objective:
            disc_output = self._get_discriminator_output(
                fake_data.inputs, self.discriminator, fake_data.is_fake_tokens)
            total_loss += config.disc_weight * disc_output.loss

        # Evaluation inputs
        eval_fn_inputs = {
            "input_ids": masked_inputs.input_ids,
            "masked_lm_preds": mlm_output.preds,
            "mlm_loss": mlm_output.per_example_loss,
            "masked_lm_ids": masked_inputs.masked_lm_ids,
            "masked_lm_weights": masked_inputs.masked_lm_weights,
            "input_mask": masked_inputs.input_mask
        }
        if config.electra_objective:
            eval_fn_inputs.update({
                "disc_loss": disc_output.per_example_loss,
                "disc_labels": disc_output.labels,
                "disc_probs": disc_output.probs,
                "disc_preds": disc_output.preds,
                "sampled_tokids": torch.argmax(fake_data.sampled_tokens, -1, ).type(torch.int64)
            })

        return total_loss, eval_fn_inputs

    def _get_masked_lm_output(self, inputs, generator):
        """Masked language modeling softmax layer."""
        masked_lm_weights = inputs.masked_lm_weights

        if self._config.uniform_generator:
            device = inputs.masked_lm_ids.device
            logits = torch.zeros(self.disc_config.vocab_size, device=device)
            logits_tiled = torch.zeros(
                list(inputs.masked_lm_ids.size()) + [self.disc_config.vocab_size], device=device)
            logits_tiled += torch.reshape(logits, [1, 1, self.disc_config.vocab_size])
            logits = logits_tiled
        else:
            outputs = generator(
                input_ids=inputs.input_ids,
                attention_mask=inputs.input_mask,
                token_type_ids=inputs.segment_ids)
            logits = outputs[0]
            logits = pretrain_utils.gather_positions(
                logits, inputs.masked_lm_positions)

        oh_labels = F.one_hot(
            inputs.masked_lm_ids, num_classes=self.disc_config.vocab_size).type(torch.float32)

        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        label_log_probs = -torch.sum(log_probs * oh_labels, dim=-1)

        numerator = torch.sum(inputs.masked_lm_weights * label_log_probs)
        denominator = torch.sum(masked_lm_weights) + 1e-6
        loss = numerator / denominator
        preds = torch.argmax(log_probs, dim=-1).type(torch.int64)

        MLMOutput = collections.namedtuple(
            "MLMOutput", ["logits", "probs", "loss", "per_example_loss", "preds"])
        return MLMOutput(
            logits=logits, probs=probs, per_example_loss=label_log_probs,
            loss=loss, preds=preds)

    def _get_discriminator_output(self, inputs, discriminator, labels):
        """Discriminator binary classifier."""
        outputs = discriminator(
            input_ids=inputs.input_ids,
            attention_mask=inputs.input_mask,
            token_type_ids=inputs.segment_ids)
        logits = outputs[0]
        weights = inputs.input_mask.type(torch.float32)
        labelsf = labels.type(torch.float32)
        loss_fct = nn.BCEWithLogitsLoss(reduction='none')
        losses = loss_fct(logits, labelsf) * weights
        per_example_loss = (torch.sum(losses, dim=-1) /
                            (1e-6 + torch.sum(weights, dim=-1)))
        loss = torch.sum(losses) / (1e-6 + torch.sum(weights))
        probs = torch.sigmoid(logits)
        preds = (torch.round((torch.sign(logits) + 1) / 2)).type(torch.int64)
        DiscOutput = collections.namedtuple(
            "DiscOutput", ["loss", "per_example_loss", "probs", "preds",
                           "labels"])
        return DiscOutput(
            loss=loss, per_example_loss=per_example_loss, probs=probs,
            preds=preds, labels=labels,
        )

    def _get_fake_data(self, inputs, mlm_logits):
        """Sample from the generator to create corrupted input."""
        inputs = pretrain_utils.unmask(inputs)
        sampled_tokens = (pretrain_utils.sample_from_softmax(
            mlm_logits / self._config.temperature)).detach()
        sampled_tokids = torch.argmax(sampled_tokens, -1).type(torch.int64)
        updated_input_ids, masked = pretrain_utils.scatter_update(
            inputs.input_ids, sampled_tokids, inputs.masked_lm_positions)
        labels = masked * (1 - updated_input_ids.eq(inputs.input_ids).type(torch.int64))
        updated_inputs = pretrain_utils.get_updated_inputs(
            inputs, input_ids=updated_input_ids)
        FakedData = collections.namedtuple("FakedData", [
            "inputs", "is_fake_tokens", "sampled_tokens"])
        return FakedData(inputs=updated_inputs, is_fake_tokens=labels,
                         sampled_tokens=sampled_tokens)


def metric_fn(config, metrics, eval_fn_inputs):
    """Computes the loss and accuracy of the model."""
    d = eval_fn_inputs
    metrics["masked_lm_accuracy"].update(
        y_true=d["masked_lm_ids"].view([-1]),
        y_pred=d["masked_lm_preds"].view([-1]),
        sample_weight=d["masked_lm_weights"].view([-1]))
    metrics["masked_lm_loss"].update(
        values=d["mlm_loss"].view([-1]),
        sample_weight=d["masked_lm_weights"].view([-1]))
    if config.electra_objective:
        metrics["sampled_masked_lm_accuracy"].update(
            y_true=d["masked_lm_ids"].view([-1]),
            y_pred=d["sampled_tokids"].view([-1]),
            sample_weight=d["masked_lm_weights"].view([-1]))
        if config.disc_weight > 0:
            metrics["disc_loss"].update(d["disc_loss"])
            metrics["disc_accuracy"].update(
                y_true=d["disc_labels"], y_pred=d["disc_preds"],
                sample_weight=d["input_mask"])
            metrics["disc_precision"].update(
                y_true=d["disc_labels"], y_pred=d["disc_preds"],
                sample_weight=d["disc_preds"] * d["input_mask"])
            metrics["disc_recall"].update(
                y_true=d["disc_labels"], y_pred=d["disc_preds"],
                sample_weight=d["disc_labels"] * d["input_mask"])
    return metrics


def get_generator_config(config, bert_config):
    """Get model config for the generator network."""
    gen_config = ElectraConfig.from_dict(bert_config.to_dict())
    gen_config.hidden_size = int(round(
        bert_config.hidden_size * config.generator_hidden_size))
    gen_config.num_hidden_layers = int(round(
        bert_config.num_hidden_layers * config.generator_layers))
    gen_config.intermediate_size = 4 * gen_config.hidden_size
    gen_config.num_attention_heads = max(1, gen_config.hidden_size // 64)
    return gen_config


def set_seed(args):
    random.seed(args.seed + get_rank())
    np.random.seed(args.seed + get_rank())
    torch.manual_seed(args.seed + get_rank())
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed + get_rank())


def train_one_step(config, model, optimizer, scheduler, features, local_step, clip_norm=1.0):
    if config.amp:
        with torch.cuda.amp.autocast():
            total_loss, eval_fn_inputs = model(features)
            if config.n_gpu > 1:
                total_loss = total_loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if config.gradient_accumulation_steps > 1:
                total_loss = total_loss / config.gradient_accumulation_steps
    else:
        total_loss, eval_fn_inputs = model(features)
        if config.n_gpu > 1:
            total_loss = total_loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
        if config.gradient_accumulation_steps > 1:
            total_loss = total_loss / config.gradient_accumulation_steps
    loss = total_loss

    if local_step % config.gradient_accumulation_steps == 0:
        if config.amp:
            scaler.scale(total_loss).backward()
            if config.optimizer.lower() == "adam":
                # Unscales the gradients of optimizer's assigned params in-place
                scaler.unscale_(optimizer)
                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scheduler.step()  # Update learning rate schedule
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if config.optimizer.lower() == "adam":
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scheduler.step()  # Update learning rate schedule
            optimizer.step()

        # model.zero_grad()
        for param in model.parameters():
            param.grad = None
    else:
        with model.no_sync():
            if config.amp:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

    return loss, eval_fn_inputs


def load_checkpoint(config, model, optimizer, scheduler, dataset_iterator):
    # Set up model checkpoint
    checkpoints = sorted(glob.glob(os.path.join(config.checkpoints_dir, "ckpt-*")),
                         key=lambda x: ("-p2-" in x, int(x.split("-")[-1].split("/")[0])))
    logger.info("** Current saved checkpoints: {}".format(str(checkpoints)))
    if config.restore_checkpoint == "latest":
        config.restore_checkpoint = checkpoints[-1] \
            if len(checkpoints) > 0 else None

    phase2_ckpt = False
    if config.restore_checkpoint is not None and os.path.isfile(
            os.path.join(config.restore_checkpoint, "checkpoint.pt")):

        checkpoint_suffix = config.restore_checkpoint.split("-")[-1].split("/")[0]
        step = int(checkpoint_suffix) + 1
        phase2_ckpt = "-p2-" in config.restore_checkpoint

        checkpoint = torch.load(os.path.join(config.restore_checkpoint, "checkpoint.pt"), map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.last_epoch = optimizer.param_groups[0]["step"] if "step" in optimizer.param_groups[0] else 0

        logger.info("** Restored model checkpoint from {}".format(config.restore_checkpoint))
    else:
        step = 0
        logger.info("** Initializing from scratch.")

    restore_iterator = bool(config.restore_checkpoint)
    # Initialize global step for phase2
    if config.phase2 and not phase2_ckpt:
        for group in optimizer.param_groups:
            group['step'] = 0
            group['lr'] = 0.0
            group['initial_lr'] = config.learning_rate
        scheduler.last_epoch = 0
        step = 0
        restore_iterator = False

    # Set up iterator checkpoint
    if restore_iterator and len(checkpoint['dataset_iterator']) == get_world_size():
        dataset_iterator.index = checkpoint["dataset_iterator"][get_rank()]
        logger.info("** Restored iterator checkpoint from {}".format(config.restore_checkpoint))

    return step, checkpoints


def save_checkpoint(config, checkpoints, model, optimizer, dataset_iterator, step):
    output_dir = os.path.join(config.checkpoints_dir, "ckpt{}-{}".format('-p2' if config.phase2 else '', step))

    dataset_index = torch.IntTensor([dataset_iterator.state_dict()['file_index']]).cuda()
    dataset_indices = [torch.IntTensor([0]).cuda() for _ in range(get_world_size())]
    torch.distributed.all_gather(dataset_indices, dataset_index)

    if is_main_process():
        utils.mkdir(output_dir)
        checkpoints.append(output_dir)
        while len(checkpoints) > config.keep_checkpoint_max:
            ckpt_to_be_removed = checkpoints.pop(0)
            utils.rmrf(ckpt_to_be_removed)


        torch.save({"model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "dataset_iterator": dataset_indices
                    },
                   os.path.join(output_dir, "checkpoint.pt"))

def main():
    # Parse essential argumentss
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="/results", help="Saving directory for checkpoints and logs")
    parser.add_argument("--model_name", required=True, help="Model name, used as the name of results' folder.")
    parser.add_argument("--data_dir", required=True, help="Path to dataset.")
    parser.add_argument("--pretrain_hdf5", type=str, help="hdf5 files used for pretraining.")

    parser.add_argument("--phase2", action='store_true',
                        help="Specified if training on phase 2 only. If not specified, default pretraining is on phase 1.")
    parser.add_argument("--seed", default=42, type=int,
                        help="Random seed.")

    parser.add_argument(
        "--amp",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--amp_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )

    parser.add_argument("--num_train_steps", type=int, help="Total number of training steps to perform.")
    parser.add_argument("--num_warmup_steps", type=int, help="Number of steps to warm up.")
    parser.add_argument("--learning_rate", type=float, help="The initial learning rate for optimizers.")
    parser.add_argument("--train_batch_size", type=int, help="Per gpu batch size for training.")
    parser.add_argument("--max_seq_length", type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")

    parser.add_argument("--mask_prob", type=float,
                        help="Percentage of input tokens to mask out / replace.")
    parser.add_argument("--disc_weight", type=float,
                        help="Ratio of discriminator loss over generator loss.")
    parser.add_argument("--generator_hidden_size", type=float,
                        help="Fraction of discriminator hidden size for generator.")

    parser.add_argument("--log_freq", type=int, default=100, help="Log every X updates steps.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of Gradient Accumulation steps")
    parser.add_argument("--save_checkpoints_steps", type=int,
                        help="Checkpoints saving frequency.")
    parser.add_argument("--keep_checkpoint_max", type=int,
                        help="Maximum number of checkpoints to keep.")
    parser.add_argument("--restore_checkpoint", default=None, type=str,
                        help="Whether to restore from a checkpoint; if specified, set to `path-to-checkpoint` or `latest`")

    parser.add_argument("--optimizer", default="adam", type=str, help="`adam` or `lamb`")
    parser.add_argument("--skip_adaptive", action='store_true',
                        help="Whether to apply adaptive LR on LayerNorm and biases")
    parser.add_argument("--lr_decay_power", type=float, default=1.0, help="LR decay power")
    parser.add_argument("--opt_beta_1", type=float, default=0.9, help="Optimizer beta1")
    parser.add_argument("--opt_beta_2", type=float, default=0.999, help="Optimizer beta2")
    parser.add_argument("--end_lr", type=float, default=0.0, help="Ending LR")

    parser.add_argument("--vocab_padding", action='store_true',
                        help="Whether to apply vocab padding")

    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")

    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1

    # Set seed
    set_seed(args)

    config = PretrainingConfig(**args.__dict__)
    args.log_dir = config.log_dir

    # DLLogger
    setup_logger(args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if get_rank() in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process global rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        get_rank(),
        device,
        args.n_gpu,
        bool(get_rank() != -1),
        config.amp,
    )

    # Set up config cont'
    if config.phase2 and not config.restore_checkpoint:
        raise ValueError("`phase2` cannot be used without `restore_checkpoint`.")

    # Padding for divisibility by 8
    if args.vocab_padding and config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
    utils.heading("Config:")
    utils.log_config(config)

    # Save pretrain configs
    pretrain_config_json = os.path.join(config.checkpoints_dir, 'pretrain_config.json')
    if is_main_process():
        utils.write_json(config.__dict__, pretrain_config_json)
        logger.info("Configuration saved in {}".format(pretrain_config_json))

    # Set up model
    model = PretrainingModel(config)
    model.to(device)

    # Set up optimizer and scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay_rate,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]

    if config.optimizer.lower() == "lamb":
        optimizer = FusedLAMB(optimizer_grouped_parameters,
                              lr=config.learning_rate,
                              betas=(config.opt_beta_1, config.opt_beta_2),
                              eps=1e-6, bias_correction=False,
                              max_grad_norm=1.0,
                              use_nvlamb=not config.skip_adaptive)
    elif config.optimizer.lower() == "adam":
        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=config.learning_rate,
                              betas=(config.opt_beta_1, config.opt_beta_2),
                              eps=1e-6, bias_correction=False)
    else:
        raise ValueError("Not supported optimizer type.")

    # optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    # if config.amp:
    #     model, optimizer = amp.initialize(model, optimizer, opt_level=config.amp_opt_level)
    scheduler = get_poly_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=config.num_train_steps,
        decay_power=config.lr_decay_power,
        end_ratio=config.end_lr / config.learning_rate
    )

    # multi-gpu training (should be after apex fp16 initialization)
    if config.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Set up dataset
    dataset_iterator = pretrain_utils.DatasetIterator(
        config, batch_size=config.train_batch_size * config.n_gpu,
        world_size=get_world_size(), rank=get_rank())


    # Set up metrics
    metrics = {}
    metrics["train_perf"] = pretrain_utils.Mean(name="train_perf")
    metrics["total_loss"] = pretrain_utils.Mean(name="total_loss")
    metrics["masked_lm_accuracy"] = pretrain_utils.Accuracy(name="masked_lm_accuracy")
    metrics["masked_lm_loss"] = pretrain_utils.Mean(name="masked_lm_loss")
    if config.electra_objective:
        metrics["sampled_masked_lm_accuracy"] = pretrain_utils.Accuracy(name="sampled_masked_lm_accuracy")
        if config.disc_weight > 0:
            metrics["disc_loss"] = pretrain_utils.Mean(name="disc_loss")
            metrics["disc_accuracy"] = pretrain_utils.Accuracy(name="disc_accuracy")
            metrics["disc_precision"] = pretrain_utils.Accuracy(name="disc_precision")
            metrics["disc_recall"] = pretrain_utils.Accuracy(name="disc_recall")

    # Set up tensorboard
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(config.log_dir, current_time,
                                 'train_' + str(get_rank()) + '_of_' + str(get_world_size()))
    train_summary_writer = SummaryWriter(train_log_dir)

    step, checkpoints = load_checkpoint(config, model, optimizer, scheduler, dataset_iterator)

    logger.info("***** Running training {}*****".format("(second phase) " if config.phase2 else ""))
    logger.info("  Instantaneous batch size per GPU = %d", config.train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        config.train_batch_size
        * config.gradient_accumulation_steps
        * get_world_size(),
    )
    logger.info("  Gradient Accumulation steps = %d", config.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", config.num_train_steps)

    model.train()
    local_step = 0
    train_start, start_step = time.time(), step - 1
    while step <= config.num_train_steps:
        for dataloader in dataset_iterator:
            if step > config.num_train_steps:
                break
            for batch in dataloader:
                batch = tuple(t.to(device) for t in batch)
                features = {
                    "input_ids": batch[0],
                    "input_mask": batch[1],
                    "segment_ids": batch[2],
                }

                local_step += 1
                iter_start = time.time()

                total_loss, eval_fn_inputs = train_one_step(config, model, optimizer, scheduler, features, local_step)

                metrics["train_perf"].update(
                    config.train_batch_size * get_world_size() / (time.time() - iter_start))
                metrics["total_loss"].update(values=total_loss)
                metric_fn(config, metrics, eval_fn_inputs)

                if local_step % config.gradient_accumulation_steps == 0:
                    # Sync up optimizer step, scheduler step and global step (later)
                    opt_step = optimizer.param_groups[0]["step"] if "step" in optimizer.param_groups[0] else 0
                    scheduler.last_epoch = opt_step

                    if config.log_freq > 0 and step != opt_step and (
                            step % config.log_freq == 0 or step == config.num_train_steps):
                        log_info_dict = {k:v.result() for k, v in metrics.items()}
                        dllogger.log(step=step, data=log_info_dict, verbosity=0)
                        print(
                            'Step:{step:6d}, Loss:{total_loss:10.6f}, Gen_loss:{masked_lm_loss:10.6f}, Disc_loss:{'
                            'disc_loss:10.6f}, Gen_acc:{masked_lm_accuracy:6.2f}, '
                            'Disc_acc:{disc_accuracy:6.2f}, Perf:{train_perf:4.0f}, Loss Scaler: {loss_scale}, '
                            'Elapsed: {elapsed}, ETA: {eta}'.format(
                                step=step,
                                **log_info_dict,
                                loss_scale=scaler.get_scale(),
                                elapsed=utils.get_readable_time(time.time() - train_start),
                                eta=utils.get_readable_time(
                                    (time.time() - train_start) / (step - start_step) * (config.num_train_steps - step)))
                        )

                        # # Last step summary
                        # if step == config.num_train_steps:
                        #     final_metrics = {}
                        #     for key, v in log_info_dict.items():
                        #         val = torch.tensor(v, device=device)
                        #         torch.distributed.all_reduce(val, op=torch.distributed.ReduceOp.SUM)
                        #         val /= get_world_size()
                        #         final_metrics[key] = val.item()
                        #     dllogger.log(step=(), data=log_info_dict, verbosity=0)
                        #     logger.info(
                        #         '<FINAL STEP METRICS> Step:{step:6d}, Loss:{total_loss:10.6f}, Gen_loss:{masked_lm_loss:10.6f}, Disc_loss:{disc_loss:10.6f}, Gen_acc:{masked_lm_accuracy:6.2f}, '
                        #         'Disc_acc:{disc_accuracy:6.2f}, Perf:{train_perf:4.0f},'.format(
                        #             step=step, **final_metrics))

                        for key, m in metrics.items():
                            train_summary_writer.add_scalar(key, m.result(), step)
                        train_summary_writer.add_scalar("lr", scheduler.get_last_lr()[0], step)

                        for m in metrics.values():
                            m.reset()

                        dllogger.flush()

                    # if config.save_checkpoints_steps > 0 and step != opt_step and \
                    #         ((step % config.save_checkpoints_steps == 0 and step > 0) or step == config.num_train_steps):
                    #     save_checkpoint(config, checkpoints, model, optimizer, dataset_iterator, step)
                    #     logger.info(f" ** Saved model checkpoint for step {step}")

                    step = opt_step
                if step > config.num_train_steps:
                    break

    train_summary_writer.flush()
    train_summary_writer.close()
    postprocess_dllog(args)


if __name__ == "__main__":
    main()
