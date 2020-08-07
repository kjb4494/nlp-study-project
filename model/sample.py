import torch
from model.model_utils import sample_gaussian


class Sample:
    _is_set_from_recog_network = False
    latent_samples = None

    # Sample data from recognized network
    recog_mu = None
    recog_logvar = None
    recog_mulogvar = None
    ctrl_latent_samples = None
    ctrl_recog_mus = None
    ctrl_recog_logvars = None
    ctrl_recog_mulogvars = None
    attribute_embedding = None
    ctrl_attribute_embeddings = None

    # Sample data from prior network
    prior_mulogvar = None
    prior_mu = None
    prior_logvar = None

    def __init__(self, s_info, f_info):
        self.s_info = s_info
        self.f_info = f_info

    def set_from_recog_network(self, cond_embedding, output_embedding):
        if self.s_info.use_hcf:
            attribute_embedding = self.s_info.da_embedding(self.f_info.out_das)
            attribute_fc1 = self.s_info.attribute_fc1(attribute_embedding)
            ctrl_attribute_embeddings = {
                da: self.s_info.da_embedding(
                    torch.ones(self.f_info.local_batch_size, dtype=torch.long, device=self.s_info.device) * idx)
                for idx, da in enumerate(self.s_info.da_vocab)
            }
            ctrl_attribute_fc1 = {
                k: self.s_info.attribute_fc1(v)
                for k, v in ctrl_attribute_embeddings.items()
            }
            recog_input = torch.cat([cond_embedding, output_embedding, attribute_fc1], 1)
            ctrl_recog_inputs = {
                k: torch.cat([cond_embedding, output_embedding, v], 1)
                for k, v in ctrl_attribute_fc1.items()
            } if self.f_info.is_train_multiple else {}
        else:
            attribute_embedding = None
            ctrl_attribute_embeddings = None
            recog_input = torch.cat([cond_embedding, output_embedding], 1)
            ctrl_recog_inputs = {
                da: torch.cat([cond_embedding, output_embedding], 1)
                for idx, da in enumerate(self.s_info.da_vocab)
            } if self.f_info.is_train_multiple else {}
        recog_mulogvar = self.s_info.recog_mulogvar_net(recog_input)
        recog_mu, recog_logvar = torch.chunk(recog_mulogvar, 2, 1)

        ctrl_latent_samples = {}
        ctrl_recog_mus = {}
        ctrl_recog_logvars = {}
        ctrl_recog_mulogvars = {}

        if self.f_info.is_train_multiple:
            latent_samples = [sample_gaussian(recog_mu, recog_logvar) for _ in range(self.f_info.num_samples)]
            ctrl_recog_mulogvars = {
                k: self.s_info.recog_mulogvar_net(v) for k, v in ctrl_recog_inputs.items()
            }
            for k in ctrl_recog_mulogvars.keys():
                ctrl_recog_mus[k], ctrl_recog_logvars[k] = torch.chunk(ctrl_recog_mulogvars[k], 2, 1)
            ctrl_latent_samples = {
                k: sample_gaussian(ctrl_recog_mus[k], ctrl_recog_logvars[k])
                for k in ctrl_recog_mulogvars.keys()
            }
        else:
            latent_samples = [sample_gaussian(recog_mu, recog_logvar)]

        self.latent_samples = latent_samples,
        self.recog_mu = recog_mu,
        self.recog_logvar = recog_logvar,
        self.recog_mulogvar = recog_mulogvar,
        self.ctrl_latent_samples = ctrl_latent_samples,
        self.ctrl_recog_mus = ctrl_recog_mus,
        self.ctrl_recog_logvars = ctrl_recog_logvars,
        self.ctrl_recog_mulogvars = ctrl_recog_mulogvars,
        self.attribute_embedding = attribute_embedding,
        self.ctrl_attribute_embeddings = ctrl_attribute_embeddings

        self._is_set_from_recog_network = True

    def set_from_prior_network(self, cond_embedding):
        self.prior_mulogvar = self.s_info.prior_mulogvar_net(cond_embedding)
        self.prior_mu, self.prior_logvar = torch.chunk(self.prior_mulogvar, 2, 1)
        if not self._is_set_from_recog_network:
            self.latent_samples = [
                sample_gaussian(self.prior_mu, self.prior_logvar) for _ in range(self.f_info.num_samples)
            ]
