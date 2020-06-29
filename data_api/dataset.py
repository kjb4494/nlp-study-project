
import torch
from torch.utils.data import Dataset
import numpy as np


class CVAEDataset(Dataset):
    def __init__(self, name, data, meta_data, config):
        # name: 데이터셋 이름
        # data: 대화 데이터 리스트
        # meta_data: 메타 데이터 리스트
        # config: 데이터셋에 대한 설정 객체
        self.name = name
        self.data_lens = [len(line) for line in data]
        self.pad_token_idx = 0
        self.utt_per_case = config['utt_per_case']
        self.max_tokenized_sent_size = config['max_utt_len']
        self.exist_inference = config.get('inference', False)
        self.indexes = list(np.argsort(self.data_lens))
        self.data = []
        self.meta_data = []

        for data_point_idx, data_point in enumerate(data):
            data_lens = len(data_point)
            meta_row = meta_data[data_point_idx]
            vec_a_meta, vec_b_meta, topic = meta_row
            if self.exist_inference:
                end_idx_offset_start = data_lens
                end_idx_offset_end = data_lens + 1
            else:
                end_idx_offset_start = 2
                # 각 대화 데이터 수
                end_idx_offset_end = data_lens

            for end_idx_offset in range(end_idx_offset_start, end_idx_offset_end):
                data_item = {}
                start_idx = max(0, end_idx_offset - self.utt_per_case)
                end_idx = end_idx_offset
                cut_row = data_point[start_idx:end_idx]
                in_row = cut_row[0:-1]
                out_row = cut_row[-1]
                vec_tokenized_sent, vec_caller, vec_senti_label = out_row
                data_item['topics'] = torch.LongTensor([topic])
                context_utts = np.zeros((self.utt_per_case, self.max_tokenized_sent_size))
                padded_utt_pairs = [self.slice_and_pad(vec_tokenized_sent)]

    def slice_and_pad(self, vec_tokenized_sent, do_pad=True):
        utt_size = len(vec_tokenized_sent)
        if utt_size >= self.max_tokenized_sent_size:
            return vec_tokenized_sent[self.max_tokenized_sent_size-1] + [vec_tokenized_sent[-1]], self.max_tokenized_sent_size
        elif do_pad:
            return vec_tokenized_sent + [self.pad_token_idx] * (self.max_tokenized_sent_size-utt_size), utt_size
        else:
            return vec_tokenized_sent, utt_size
