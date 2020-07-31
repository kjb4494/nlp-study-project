import torch

from model.cvae_feed_info import CVAEFeedInfo
from model.model_utils import sample_gaussian


class Sample:
    def __init__(self, info: CVAEFeedInfo):
        self.f_info = info

    def set_sample_from_recog_network(self, cond_embedding, output_embedding):
        if self.f_info.info.use_hcf:
            attribute_embedding = self.f_info.info.da_embedding(self.f_info.out_das)
            attribute_fc1 = self.f_info.info.attribute_fc1(attribute_embedding)
            ctrl_attribute_embeddings = {
                da: self.f_info.info.da_embedding(
                    torch.ones(self.f_info.local_batch_size, dtype=torch.long, device=self.f_info.info.device) * idx)
                for idx, da in enumerate(self.f_info.info.da_vocab)
            }
            ctrl_attribute_fc1 = {
                k: self.f_info.info.attribute_fc1(v)
                for k, v in ctrl_attribute_embeddings.items()
            }
            recog_input = torch.cat([cond_embedding, output_embedding, attribute_fc1], 1)
            ctrl_recog_inputs = {
                k: torch.cat([cond_embedding, self.output_embedding, v], 1)
                for k, v in ctrl_attribute_fc1.items()
            } if self.f_info.is_train_multiple else {}
        else:
            attribute_embedding = None
            ctrl_attribute_embeddings = None
            recog_input = torch.cat([self.cond_embedding, self.output_embedding], 1)
            ctrl_recog_inputs = {
                da: torch.cat([self.cond_embedding, self.output_embedding], 1)
                for idx, da in enumerate(self.f_info.info.da_vocab)
            } if self.f_info.is_train_multiple else {}
        recog_mulogvar = self.f_info.info.recog_mulogvar_net(recog_input)
        recog_mu, recog_logvar = torch.chunk(recog_mulogvar, 2, 1)

        ctrl_latent_samples = {}
        ctrl_recog_mus = {}
        ctrl_recog_logvars = {}
        ctrl_recog_mulogvars = {}

        if self.f_info.is_train_multiple:
            latent_samples = [sample_gaussian(recog_mu, recog_logvar) for _ in range(self.f_info.num_samples)]
            ctrl_recog_mulogvars = {
                k: self.f_info.info.recog_mulogvar_net(v) for k, v in ctrl_recog_inputs.items()
            }
            for k in ctrl_recog_mulogvars.keys():
                ctrl_recog_mus[k], ctrl_recog_logvars[k] = torch.chunk(ctrl_recog_mulogvars[k], 2, 1)
            ctrl_latent_samples = {
                k: sample_gaussian(ctrl_recog_mus[k], ctrl_recog_logvars[k])
                for k in ctrl_recog_mulogvars.keys()
            }
        else:
            latent_samples = [sample_gaussian(recog_mu, recog_logvar)]

        sample_result = {
            'latent_samples': latent_samples,
            'recog_mu': recog_mu,
            'recog_logvar': recog_logvar,
            'recog_mulogvar': recog_mulogvar,
            'ctrl_latent_samples': ctrl_latent_samples,
            'ctrl_recog_mus': ctrl_recog_mus,
            'ctrl_recog_logvars': ctrl_recog_logvars,
            'ctrl_recog_mulogvars:': ctrl_recog_mulogvars,
            'attribute_embedding': attribute_embedding,
            'ctrl_attribute_embeddings': ctrl_attribute_embeddings
        }

        return sample_result
