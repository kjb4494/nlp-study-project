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
        out_token = model_output["out_token"]
        out_das = model_output["out_das"]
        da_logit = model_output["da_logit"]
        bow_logit = model_output["bow_logit"]
        dec_out = model_output["dec_out"]

        return self.calculate_loss(dec_out, bow_logit, da_logit, out_token,
                                   out_das, model_output, is_train, is_valid, current_step)
