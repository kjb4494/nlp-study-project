
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
        self.tokenized_sent_per_case = config['tokenized_sent_per_case']
        self.max_tokenized_sent_size = config['max_tokenized_sent_size']
        self.exist_inference = config.get('inference', False)
        self.indexes = list(np.argsort(self.data_lens))
        self.data = []
        self.meta_data = []

        for data_point_idx, data_point in enumerate(data):
            data_size = len(data_point)
            meta_row = meta_data[data_point_idx]
            vec_a_meta, vec_b_meta, topic = meta_row
            if self.exist_inference:
                end_idx_offset_start = data_size
                end_idx_offset_end = data_size + 1
            else:
                end_idx_offset_start = 2
                # 각 대화 데이터 수
                end_idx_offset_end = data_size

            for end_idx_offset in range(end_idx_offset_start, end_idx_offset_end):
                start_idx = max(0, end_idx_offset - self.tokenized_sent_per_case)
                end_idx = end_idx_offset
                cut_row = data_point[start_idx:end_idx]
                in_row = cut_row[:-1]
                out_row = cut_row[-1]
                or_vec_tokenized_sent, or_vec_caller, or_vec_senti_label = out_row

                context_tokenized_sents = np.zeros((self.tokenized_sent_per_case, self.max_tokenized_sent_size))
                padded_vec_tokenized_sent_pairs = [
                    self.slice_and_pad(vec_tokenized_sent=vec_tokenized_sent) for vec_tokenized_sent, _, _ in in_row
                ]
                padded_vec_tokenized_sents = [
                    vec_tokenized_sent_pair[0] for vec_tokenized_sent_pair in padded_vec_tokenized_sent_pairs
                ]
                context_tokenized_sents[:len(in_row)] = padded_vec_tokenized_sents
                in_row_size = np.zeros(self.tokenized_sent_per_case)
                in_row_size[:len(in_row)] = [
                    vec_tokenized_sent_pair[1] for vec_tokenized_sent_pair in padded_vec_tokenized_sent_pairs
                ]
                callers = np.zeros(self.tokenized_sent_per_case)
                callers[:len(in_row)] = [
                    int(caller == or_vec_caller) for _, caller, _ in in_row
                ]
                padded_vec_tokenized_sents = self.slice_and_pad(vec_tokenized_sent=or_vec_tokenized_sent)

                data_item = {
                    'topics': torch.LongTensor([topic]),
                    'my_profile': torch.FloatTensor(vec_a_meta) if or_vec_caller == 0 else torch.FloatTensor(vec_b_meta),
                    'ot_profile': torch.FloatTensor(vec_b_meta) if or_vec_caller == 0 else torch.FloatTensor(vec_a_meta),
                    'context_lens': torch.LongTensor([len(in_row)]),
                    'context_utts': torch.LongTensor(context_tokenized_sents),
                    'floors': torch.LongTensor(callers),
                    'out_utts': torch.LongTensor(padded_vec_tokenized_sents[0]),
                    'out_lens': torch.LongTensor([padded_vec_tokenized_sents[1]]),
                    'out_floor': torch.LongTensor([or_vec_caller]),
                    'out_das': torch.LongTensor([or_vec_senti_label])
                }
                self.data.append(data_item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def slice_and_pad(self, vec_tokenized_sent, do_pad=True):
        utt_size = len(vec_tokenized_sent)
        if utt_size >= self.max_tokenized_sent_size:
            return vec_tokenized_sent[:self.max_tokenized_sent_size-1] + [vec_tokenized_sent[-1]], self.max_tokenized_sent_size
        elif do_pad:
            return vec_tokenized_sent + [self.pad_token_idx] * (self.max_tokenized_sent_size-utt_size), utt_size
        else:
            return vec_tokenized_sent, utt_size
