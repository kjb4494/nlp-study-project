import torch
import torch.nn as nn
import torch.nn.functional as fnn

import numpy as np

from model.cvae_info import CVAEModelInfo
from model.model_utils import get_bi_rnn_encode


class CVAEModel(nn.Module):
    def __init__(self, model_config, vocab_class):
        super(CVAEModel, self).__init__()
        self.info = CVAEModelInfo(model_config=model_config, vocab_class=vocab_class)

        # Tensor Value For Parameters() in Optimizer
        self.topic_embedding = self.info.topic_embedding
        self.da_embedding = self.info.da_embedding
        self.word_embedding = self.info.word_embedding
        self.bi_sent_cell = self.info.bi_sent_cell
        self.enc_cell = self.info.enc_cell
        self.attribute_fc1 = self.info.attribute_fc1
        self.recog_mulogvar_net = self.info.recog_mulogvar_net
        self.prior_mulogvar_net = self.info.prior_mulogvar_net
        self.bow_project = self.info.bow_project
        self.da_project = self.info.da_project
        self.dec_init_state_net = self.info.dec_init_state_net
        self.dec_cell = self.info.dec_cell
        self.dec_cell_project = self.info.dec_cell_project

    # arg: torch DataLoader
    def forward(self, feed_dict):
        is_train = feed_dict['is_train']
        is_train_multiple = feed_dict.get('is_train_multiple', False)
        is_test_multi_da = feed_dict.get('is_test_multi_da', True)
        num_samples = feed_dict['num_samples']

        # if is_train:
        #     model_output =

    # 리팩토링이 절실한 부분 - 리턴 데이터가 더러움
    def feed_train(self, feed_dict):
        is_train_multiple = feed_dict.get('is_train_multiple', False)
        num_samples = feed_dict['num_samples']
        context_lens, input_contexts, floors, topics, my_profile, ot_profile = self.info.get_converted_info_to_device_context(
            feed_dict)
        out_tok, out_das, output_lens = self.info.get_converted_info_to_device_output(feed_dict)
        output_embedded = self.info.word_embedding(out_tok)
        if self.info.sent_type == 'bi-rnn':
            output_embedding, _ = get_bi_rnn_encode(output_embedded, self.info.bi_sent_cell,
                                                    self.info.max_tokenized_sent_size)
        else:
            raise ValueError("unk sent_type. select one in [bow, rnn, bi-rnn]")
        relation_embedded = self.info.topic_embedding(topics)
        local_batch_size = input_contexts.size(0)
        max_dialog_len = input_contexts.size(1)
        max_seq_len = input_contexts.size(-1)

        enc_last_state = self.info.get_encoder_state(
            input_contexts=input_contexts,
            floors=floors,
            is_train=True,
            context_lens=context_lens,
            max_dialog_len=max_dialog_len,
            max_seq_len=max_seq_len
        )

        cond_list = [relation_embedded, my_profile, ot_profile, enc_last_state]
        cond_embedding = torch.cat(cond_list, 1)

        sample_result = self.info.get_sample_from_recog_network(
            local_batch_size=local_batch_size,
            cond_embedding=cond_embedding,
            num_samples=num_samples,
            out_das=out_das,
            output_embedding=output_embedding,
            is_train_multiple=is_train_multiple
        )

        latent_samples, recog_mu, recog_logvar, recog_mulogvar, \
        ctrl_latent_samples, ctrl_recog_mus, ctrl_recog_logvars, ctrl_recog_mulogvars, \
        attribute_embedding, ctrl_attribute_embeddings = sample_result

        prior_sample_result = self._sample_from_prior_network(cond_embedding, num_samples)
        _, prior_mu, prior_logvar, prior_mulogvar = prior_sample_result

        def get_dec_input_test():
            gen_inputs = [torch.cat([cond_embedding, latent_sample], 1) for latent_sample in latent_samples]
            bow_logit = self.bow_project(gen_inputs[0])
            if self.info.use_hcf:
                da_logits = [self.da_project(gen_input) for gen_input in gen_inputs]
                da_probs = [fnn.softmax(da_logits, dim=1) for da_logit in da_logits]
                pred_attribute_embeddings = [torch.matmul(da_prob, self.da_embedding.weight) for da_prob in da_probs]
                dec_inputs = [
                    torch.cat((gen_input, pred_attribute_embeddings[i]), 1) for i, gen_input in enumerate(gen_inputs)
                ]
            else:
                da_logits = [gen_input.new_zeros(local_batch_size, self.info.da_size) for gen_input in gen_inputs]
                dec_inputs = gen_inputs

            # decoder
            if self.info.num_layer > 1:
                dec_init_states = [
                    [self.info.dec_init_state_net[i](dec_input) for i in range(self.info.num_layer)]
                    for dec_input in dec_inputs
                ]
                dec_init_states = [torch.stack(dec_init_state) for dec_init_state in dec_init_states]
            else:
                dec_init_states = [self.info.dec_init_state_net(dec_input).unsqueeze(0) for dec_input in dec_inputs]

            return da_logits, bow_logit, dec_inputs, dec_init_states, pred_attribute_embeddings

        da_logits, bow_logit, dec_inputs, dec_init_states, pred_attribute_embeddings = get_dec_input_test()
