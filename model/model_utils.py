import numpy as np

import torch.nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch
import torch.nn as nn


def sample_gaussian(mu, logvar):
    epsilon = logvar.new_empty(logvar.size()).normal_()
    std = torch.exp(0.5 * logvar)
    z = mu + std * epsilon
    return z


def dynamic_rnn(cell, inputs, sequence_length, max_len, init_state=None, output_fn=None):
    sorted_lens, len_ix = sequence_length.sort(0, descending=True)
    inv_ix = len_ix.clone()
    inv_ix[len_ix] = torch.arange(0, len(len_ix)).type_as(inv_ix)

    valid_num = torch.sign(sorted_lens).long().sum().item()
    zero_num = inputs.size(0) - valid_num

    # 이 부분 의미가 뭐냐?
    sorted_inputs = inputs[len_ix].contiguous()
    if init_state is not None:
        sorted_init_state = init_state[:, len_ix].contiguous()

    return None, None


def get_bi_rnn_encode(embedding, cell, max_len, length_mask=None):
    if length_mask is None:
        length_mask = torch.sum(torch.sign(torch.max(torch.abs(embedding), 2)[0]), 1)
        length_mask = length_mask.long()
    _, encoded_input = dynamic_rnn(
        cell=cell,
        inputs=embedding,
        sequence_length=length_mask, max_len=max_len
    )
    encoded_input = torch.cat([encoded_input[-2], encoded_input[-1]], 1)
    return encoded_input, cell.hidden_size * 2
