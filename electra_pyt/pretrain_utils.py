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

"""Helpers for preparing pre-training data and supplying them to the model."""
import six
import glob
import h5py
import torch
import random
import collections
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import sys
import utils
import tokenization
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler, RandomSampler, SequentialSampler, Dataset

# model inputs - it's a bit nicer to use a namedtuple rather than keep the
# features as a dict
Inputs = collections.namedtuple(
    "Inputs", ["input_ids", "input_mask", "segment_ids", "masked_lm_positions",
               "masked_lm_ids", "masked_lm_weights"])


# Workaround because python functions are not picklable
class WorkerInitObj(object):
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, id):
        np.random.seed(seed=self.seed + id)
        random.seed(self.seed + id)


class DatasetIterator:
    def __init__(self, config, batch_size, world_size=1, rank=0):
        self.config = config
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.index = 0
        self.future_dataloader = None

        self.worker_init = WorkerInitObj(config.seed + rank)
        self.pool = ProcessPoolExecutor(max_workers=1)

        input_files = []
        for input_pattern in config.pretrain_hdf5.split(","):
            input_files.extend(glob.glob(input_pattern))
        self.input_files = input_files[rank::world_size]

        random.seed(config.seed)
        random.shuffle(self.input_files)

    def __iter__(self):
        self.load_future()
        return self

    def __next__(self):
        dataloader = self.future_dataloader.result(timeout=None)
        self.index += 1
        if self.index >= len(self.input_files):
            self.index = 0
            random.shuffle(self.input_files)
        self.load_future()
        return dataloader

    def load_future(self):
        self.future_dataloader = self.pool.submit(
            create_dataset,
            self.input_files[self.index],
            self.config.max_seq_length,
            self.batch_size,
            self.worker_init
        )

    def load_state_dict(self, state_dict):
        self.index = state_dict['file_index']

    def state_dict(self):
        return {
            'file_index': self.index - 1, # We want to point to the current dataloader, not a future one
        }



def create_dataset(input_file, max_seq_length, batch_size, worker_init, num_cpu_threads=4):
    dataset = PretrainDataset(
        input_file=input_file, max_pred_length=max_seq_length)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_cpu_threads,
        worker_init_fn=worker_init,
        drop_last=True,
        pin_memory=False)
    return dataloader


class PretrainDataset(Dataset):
    def __init__(self, input_file, max_pred_length):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        keys = ["input_ids", "input_mask", "segment_ids"]
        with h5py.File(input_file, "r") as f:
            self.dataset = {key: torch.tensor(f[key], dtype=torch.int64)
                            for key in keys}

    def __len__(self):
        return len(self.dataset["input_ids"])

    def __getitem__(self, index):
        return self.dataset["input_ids"][index], \
               self.dataset["input_mask"][index], \
               self.dataset["segment_ids"][index]


class Accuracy:
    def __init__(self, **kwargs):
        self._num_correct = 0
        self._num_examples = 0

    def reset(self):
        self._num_correct = 0
        self._num_examples = 0

    def update(self, y_true, y_pred, sample_weight=None):
        if sample_weight is None:
            correct = torch.sum(y_pred.eq(y_true).int()).item()
            num_examples = torch.numel(y_true)
        else:
            correct = (y_pred.eq(y_true) * sample_weight).int()
            correct = torch.sum(correct).item()
            num_examples = torch.sum(sample_weight).item()

        self._num_correct += correct
        self._num_examples += num_examples

    def result(self):
        if self._num_examples == 0:
            return float("nan")
        return self._num_correct / self._num_examples


class Mean:
    def __init__(self, **kwargs):
        self._total = 0.0
        self._num_examples = 0

    def reset(self):
        self._total = 0.0
        self._num_examples = 0

    def update(self, values, sample_weight=None):
        if sample_weight is None:
            if not isinstance(values, torch.Tensor):
                values = torch.tensor(values)
            if len(values.shape) == 0:
                values.unsqueeze_(-1)
            self._total += torch.sum(values).item()
            self._num_examples += values.shape[0]
        else:
            self._total += torch.sum(values * sample_weight).item()
            self._num_examples += torch.sum(sample_weight).item()

    def result(self):
        if self._num_examples == 0:
            return float("nan")
        return self._total / self._num_examples


def features_to_inputs(features):
    return Inputs(
        input_ids=features["input_ids"],
        input_mask=features["input_mask"],
        segment_ids=features["segment_ids"],
        masked_lm_positions=(features["masked_lm_positions"]
                             if "masked_lm_positions" in features else None),
        masked_lm_ids=(features["masked_lm_ids"]
                       if "masked_lm_ids" in features else None),
        masked_lm_weights=(features["masked_lm_weights"]
                           if "masked_lm_weights" in features else None),
    )


def get_updated_inputs(inputs, **kwargs):
    features = inputs._asdict()
    for k, v in kwargs.items():
        features[k] = v
    return features_to_inputs(features)


# @torch.jit.script
def gather_positions(sequence, positions):
    """Gathers the vectors at the specific positions over a minibatch.

  Args:
    sequence: A [batch_size, seq_length] or
        [batch_size, seq_length, depth] tensor of values
    positions: A [batch_size, n_positions] tensor of indices

  Returns: A [batch_size, n_positions] or
    [batch_size, n_positions, depth] tensor of the values at the indices
  """
    shape = sequence.size()
    depth_dimension = (len(shape) == 3)
    if depth_dimension:
        B, L, D = shape
    else:
        B, L = shape
        D = 1

    position_shift = torch.unsqueeze(L * torch.arange(B, device=positions.device), -1)
    flat_positions = torch.reshape(positions + position_shift, [-1])
    flat_sequence = torch.reshape(sequence, [B * L, D])
    gathered = flat_sequence[flat_positions]
    if depth_dimension:
        return torch.reshape(gathered, [B, -1, D])
    else:
        return torch.reshape(gathered, [B, -1])


# @torch.jit.script
def scatter_update(sequence, updates, positions):
    """Scatter-update a sequence.

  Args:
    sequence: A [batch_size, seq_len] or [batch_size, seq_len, depth] tensor
    updates: A tensor of size batch_size*seq_len(*depth)
    positions: A [batch_size, n_positions] tensor

  Returns: A tuple of two tensors. First is a [batch_size, seq_len] or
    [batch_size, seq_len, depth] tensor of "sequence" with elements at
    "positions" replaced by the values at "updates." Updates to index 0 are
    ignored. If there are duplicated positions the update is only applied once.
    Second is a [batch_size, seq_len] mask tensor of which inputs were updated.
  """
    shape = sequence.size()
    depth_dimension = (len(shape) == 3)
    if depth_dimension:
        B, L, D = shape
        sequence_3d = sequence
    else:
        B, L = shape
        D = 1
        sequence_3d = sequence.unsqueeze(-1)
    _, N = positions.size()

    device = sequence.device
    shift = torch.unsqueeze(L * torch.arange(B, device=device), -1)
    flat_positions = torch.reshape(positions + shift, [-1])
    flat_updates = torch.reshape(updates, [-1, D])
    updates = torch.zeros([B * L, D], dtype=updates.dtype, device=device)
    updates[flat_positions] = flat_updates
    updates = torch.reshape(updates, [B, L, D])

    flat_updates_mask = torch.ones([B * N], dtype=torch.int64, device=device)
    updates_mask = torch.zeros([B * L], dtype=torch.int64, device=device)
    updates_mask[flat_positions] = flat_updates_mask
    updates_mask = torch.reshape(updates_mask, [B, L])
    not_first_token = torch.cat([torch.zeros((B, 1), dtype=torch.int64, device=device),
                                 torch.ones((B, L - 1), dtype=torch.int64, device=device)], -1)
    updates_mask *= not_first_token
    updates_mask_3d = torch.unsqueeze(updates_mask, -1)

    # account for duplicate positions
    if sequence.dtype == torch.float32:
        updates_mask_3d = updates_mask_3d.to(torch.float32)
        updates /= torch.clamp_min(updates_mask_3d, 1.0)
    else:
        assert sequence.dtype == torch.int64
        updates = (updates // torch.clamp_min(updates_mask_3d, 1))
    updates_mask = torch.clamp_max(updates_mask, 1)
    updates_mask_3d = torch.clamp_max(updates_mask_3d, 1)

    updated_sequence = (((1 - updates_mask_3d) * sequence_3d) +
                        (updates_mask_3d * updates))
    if not depth_dimension:
        updated_sequence = torch.squeeze(updated_sequence, -1)

    return updated_sequence, updates_mask


def _get_candidates_mask(inputs: Inputs, vocab,
                         disallow_from_mask=None):
    """Returns a mask tensor of positions in the input that can be masked out."""
    ignore_ids = [vocab["[SEP]"], vocab["[CLS]"], vocab["[MASK]"]]
    candidates_mask = torch.ones_like(inputs.input_ids, dtype=torch.bool, device=inputs.input_ids.device)
    for ignore_id in ignore_ids:
        candidates_mask &= ~inputs.input_ids.eq(ignore_id)
    candidates_mask &= inputs.input_mask.type(torch.bool)
    if disallow_from_mask is not None:
        candidates_mask &= ~disallow_from_mask
    return candidates_mask


def mask(config, inputs, mask_prob,vocab, proposal_distribution=1.0,
         disallow_from_mask=None, already_masked=None):
    """Implementation of dynamic masking. The optional arguments aren't needed for
    BERT/ELECTRA and are from early experiments in "strategically" masking out
    tokens instead of uniformly at random.

    Args:
      config: configure_pretraining.PretrainingConfig
      inputs: pretrain_data.Inputs containing input input_ids/input_mask
      mask_prob: percent of tokens to mask
      proposal_distribution: for non-uniform masking can be a [B, L] tensor
                             of scores for masking each position.
      disallow_from_mask: a boolean tensor of [B, L] of positions that should
                          not be masked out
      already_masked: a boolean tensor of [B, N] of already masked-out tokens
                      for multiple rounds of masking
    Returns: a pretrain_data.Inputs with masking added
    """
    # Get the batch size, sequence length, and max masked-out tokens
    N = config.max_predictions_per_seq
    B, L = inputs.input_ids.size()

    # Find indices where masking out a token is allowed
    candidates_mask = _get_candidates_mask(inputs, vocab, disallow_from_mask)

    # Set the number of tokens to mask out per example
    num_tokens = torch.sum(inputs.input_mask, -1).type(torch.float32)
    num_to_predict = torch.clamp_min(torch.clamp_max(
        torch.round(num_tokens * mask_prob).type(torch.int64), N), 1)
    masked_lm_weights = sequence_mask(num_to_predict, N).type(torch.float32)
    if already_masked is not None:
        masked_lm_weights *= (1 - already_masked)

    # Get a probability of masking each position in the sequence
    candidate_mask_float = candidates_mask.type(torch.float32)
    sample_prob = (proposal_distribution * candidate_mask_float)
    sample_prob /= torch.sum(sample_prob, dim=-1, keepdim=True)

    # Sample the positions to mask out
    sample_prob = sample_prob.detach()
    masked_lm_positions = torch.multinomial(
        sample_prob, N, replacement=True).type(torch.int64)
    masked_lm_positions *= masked_lm_weights.type(torch.int64)
    device = masked_lm_positions.device

    # Get the ids of the masked-out tokens
    shift = torch.unsqueeze(L * torch.arange(B, device=device), -1)
    flat_positions = torch.reshape(masked_lm_positions + shift, [-1])
    masked_lm_ids = torch.reshape(inputs.input_ids, [-1])[flat_positions]
    masked_lm_ids = torch.reshape(masked_lm_ids, [B, -1])
    masked_lm_ids *= masked_lm_weights.type(torch.int64)

    # Update the input ids
    replace_with_mask_positions = masked_lm_positions * \
                                  (torch.rand([B, N], device=device) < 0.85).type(torch.int64)
    inputs_ids, _ = scatter_update(
        inputs.input_ids,
        torch.full([B, N], vocab["[MASK]"], dtype=inputs.input_ids.dtype, device=device),
        replace_with_mask_positions)
    print(inputs_ids.detach().dtype, inputs_ids.detach().size())
    print(masked_lm_positions.dtype, masked_lm_positions.detach().size())
    print(masked_lm_ids.dtype, masked_lm_ids.detach().size())
    print(masked_lm_weights.dtype, masked_lm_weights.detach().size())
    sys.exit()
    return get_updated_inputs(
        inputs,
        input_ids=inputs_ids.detach(),
        masked_lm_positions=masked_lm_positions,
        masked_lm_ids=masked_lm_ids,
        masked_lm_weights=masked_lm_weights
    )


def unmask(inputs: Inputs):
    unmasked_input_ids, _ = scatter_update(
        inputs.input_ids, inputs.masked_lm_ids, inputs.masked_lm_positions)
    return get_updated_inputs(inputs, input_ids=unmasked_input_ids)


# @torch.jit.script
def sample_from_softmax(logits):
    uniform_noise = torch.rand(logits.size(), device=logits.device)
    gumbel_noise = -torch.log(-torch.log(uniform_noise + 1e-9) + 1e-9)
    tensor_to_one_hot = torch.argmax(F.softmax(logits + gumbel_noise, dim=-1), dim=-1).to(torch.int64)
    # one_hot = F.one_hot(tensor_to_one_hot,logits.shape[-1])
    tensorshape = tensor_to_one_hot.size()
    # print(tensorshape)
    one_hot = torch.zeros(list(tensorshape) + [logits.shape[-1]], device=torch.cuda.current_device())
    one_hot.scatter_(2, tensor_to_one_hot.reshape(tensorshape[0], tensorshape[1],1), 1)
    return one_hot


# @torch.jit.script
def sequence_mask(lengths, maxlen):
    seq_mask = torch.le(torch.arange(maxlen, device=lengths.device)[None, :],
                        lengths[:, None])
    return seq_mask


ENDC = "\033[0m"
COLORS = ["\033[" + str(n) + "m" for n in list(range(91, 97)) + [90]]
RED = COLORS[0]
BLUE = COLORS[3]
CYAN = COLORS[5]
GREEN = COLORS[1]


def print_tokens(inputs: Inputs, inv_vocab, updates_mask=None):
    """Pretty-print model inputs."""
    pos_to_tokid = {}
    for tokid, pos, weight in zip(
            inputs.masked_lm_ids[0], inputs.masked_lm_positions[0],
            inputs.masked_lm_weights[0]):
        if weight == 0:
            pass
        else:
            pos_to_tokid[pos] = tokid

    text = ""
    provided_update_mask = (updates_mask is not None)
    if not provided_update_mask:
        updates_mask = np.zeros_like(inputs.input_ids)
    for pos, (tokid, um) in enumerate(
            zip(inputs.input_ids[0], updates_mask[0])):
        token = inv_vocab[tokid]
        if token == "[PAD]":
            break
        if pos in pos_to_tokid:
            token = RED + token + " (" + inv_vocab[pos_to_tokid[pos]] + ")" + ENDC
            if provided_update_mask:
                assert um == 1
        else:
            if provided_update_mask:
                assert um == 0
        text += token + " "
    utils.log(utils.printable_text(text))
