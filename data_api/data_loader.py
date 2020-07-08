
import torch
from torch.utils.data.dataloader import default_collate
import numpy as np


def get_cvae_collate(tokenized_sent_per_case, max_tokenized_sent_size):
    def cave_collate(cvae_data_list):
        batch_size = len(cvae_data_list)
        collate_result = default_collate(cvae_data_list)

        vec_context = np.zeros((batch_size, tokenized_sent_per_case, max_tokenized_sent_size))
        vec_floors = np.zeros((batch_size, tokenized_sent_per_case))
        vec_outs = np.zeros((batch_size, max_tokenized_sent_size))

        for idx, item in enumerate(cvae_data_list):
            vec_context[idx] = item['context_utts']
            vec_floors[idx] = item['floors']
            vec_outs[idx] = item['out_utts']

        collate_result['vec_context'] = torch.LongTensor(vec_context)
        collate_result['vec_floors'] = torch.LongTensor(vec_floors)
        collate_result['vec_outs'] = torch.LongTensor(vec_outs)
        return collate_result
    return cave_collate
