import torch
import torch.nn as nn
import torch.nn.functional as fnn


class CVAELoss(nn.Module):
    def __init__(self, config):
        super(CVAELoss, self).__init__()
        self.use_hcf = config['use_hcf']
        self.full_kl_step = config['full_kl_step']

    # 상속
    def forward(self, model_output, model_input, current_step, is_train, is_valid):
        pass