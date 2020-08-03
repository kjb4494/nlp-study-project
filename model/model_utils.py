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


# 상세 분석 필요함
def dynamic_rnn(cell, inputs, sequence_length, max_len, init_state=None, output_fn=None):
    sorted_lens, len_ix = sequence_length.sort(0, descending=True)
    inv_ix = len_ix.clone()
    inv_ix[len_ix] = torch.arange(0, len(len_ix)).type_as(inv_ix)

    valid_num = torch.sign(sorted_lens).long().sum().item()
    zero_num = inputs.size(0) - valid_num

    sorted_inputs = inputs[len_ix].contiguous()
    packed_inputs = pack_padded_sequence(
        sorted_inputs[:valid_num],
        list(sorted_lens[:valid_num]), batch_first=True
    )

    if init_state is not None:
        sorted_init_state = init_state[:, len_ix].contiguous()
        outputs, state = cell(packed_inputs, sorted_init_state[:, :valid_num])
        if zero_num > 0:
            state = torch.cat([state, sorted_init_state[:, valid_num]], 1)
    else:
        outputs, state = cell(packed_inputs)
        if zero_num > 0:
            state = torch.cat([state, state.new_zeros(state.size(0), zero_num, state.size(2))], 1)

    outputs, _ = pad_packed_sequence(outputs, batch_first=True, total_length=max_len)
    if zero_num > 0:
        outputs = torch.cat([outputs, outputs.new_zeros(zero_num, outputs.size(1), outputs.size(2))], 0)

    # Reorder to the original order
    new_outputs = outputs[inv_ix].contiguous()
    new_state = state[:, inv_ix].contiguous()

    # compensate the last last layer dropout, necessary????????? need to check!!!!!!!!
    new_new_state = F.dropout(new_state, cell.dropout, cell.training)
    new_new_outputs = F.dropout(new_outputs, cell.dropout, cell.training)

    if output_fn is not None:
        new_new_outputs = output_fn(new_new_outputs)

    return new_new_outputs, new_new_state


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
