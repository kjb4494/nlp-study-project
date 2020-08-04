import torch
import torch.nn as nn
import torch.nn.functional as fnn

import numpy as np

from model.cvae_static_info import CVAEStaticInfo
from model.cvae_feed_info import CVAEFeedInfo
from model.model_utils import get_bi_rnn_encode


class CVAEModel(nn.Module):
    def __init__(self, model_config, vocab_class):
        super(CVAEModel, self).__init__()
        self.s_info = CVAEStaticInfo(model_config=model_config, vocab_class=vocab_class)

        # Tensor Value For Parameters() in Optimizer
        self.topic_embedding = self.s_info.topic_embedding
        self.da_embedding = self.s_info.da_embedding
        self.word_embedding = self.s_info.word_embedding
        self.bi_sent_cell = self.s_info.bi_sent_cell
        self.enc_cell = self.s_info.enc_cell
        self.attribute_fc1 = self.s_info.attribute_fc1
        self.recog_mulogvar_net = self.s_info.recog_mulogvar_net
        self.prior_mulogvar_net = self.s_info.prior_mulogvar_net
        self.bow_project = self.s_info.bow_project
        self.da_project = self.s_info.da_project
        self.dec_init_state_net = self.s_info.dec_init_state_net
        self.dec_cell = self.s_info.dec_cell
        self.dec_cell_project = self.s_info.dec_cell_project

    # arg: torch DataLoader
    def forward(self, feed_dict):
        device = self.s_info.device
        f_info = CVAEFeedInfo(feed_dict, device)
        if f_info.is_train:
            f_info.feed_train(s_info=self.s_info)
        else:
            f_info.feed_inference(s_info=self.s_info)
