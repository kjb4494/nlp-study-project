import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as fnn

from model.model_utils import get_bi_rnn_encode, dynamic_rnn, sample_gaussian


class CVAEModelInfo:
    def __init__(self, model_config, vocab_class):
        self.vocab = vocab_class.vocab
        self.rev_vocab = vocab_class.rev_vocab
        self.vocab_size = len(self.vocab)

        self.topic_vocab = vocab_class.topic_vocab
        self.topic_vocab_size = len(self.topic_vocab)

        self.da_vocab = vocab_class.dialog_act_vocab
        self.da_vocab_size = len(self.da_vocab)

        self.pad_id = self.rev_vocab['<pad>']
        self.go_id = self.rev_vocab['<s>']
        self.eos_id = self.rev_vocab['</s>']

        self.max_tokenized_sent_size = model_config['max_tokenized_sent_size']
        self.ctx_cell_size = model_config['ctx_cell_size']
        self.sent_cell_size = model_config['sent_cell_size']
        self.dec_cell_size = model_config['dec_cell_size']
        self.latent_size = model_config['latent_size']
        self.embed_size = model_config['embed_size']
        self.sent_type = model_config['sent_type']
        self.keep_prob = model_config['keep_prob']
        self.num_layer = model_config['num_layer']
        self.use_hcf = model_config['use_hcf']
        self.device = torch.device(model_config['device'])
        self.dec_keep_prob = model_config['dec_keep_prob']
        self.topic_embed_size = model_config['topic_embed_size']
        self.da_size = model_config['da_size']
        self.da_embed_size = model_config['da_embed_size']
        self.da_hidden_size = model_config['da_hidden_size']
        self.meta_embed_size = model_config['meta_embed_size']
        self.bow_hidden_size = model_config['bow_hidden_size']
        self.act_hidden_size = model_config['act_hidden_size']

        self.topic_embedding = nn.Embedding(self.topic_vocab_size, self.topic_embed_size)
        self.da_embedding = nn.Embedding(self.da_vocab_size, self.da_embed_size)
        self.word_embedding = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=self.pad_id)

        if vocab_class.word2vec is not None:
            self.word_embedding.from_pretrained(
                torch.FloatTensor(vocab_class.word2vec),
                padding_idx=self.pad_id
            )
        if self.sent_type == 'bi-rnn':
            self.bi_sent_cell = nn.GRU(
                input_size=self.embed_size,
                hidden_size=self.sent_cell_size,
                num_layers=self.num_layer,
                dropout=1 - self.keep_prob,
                bidirectional=True
            )
        else:
            raise ValueError('Unknown sent_type... Only use bi-rnn type.')

        input_embedding_size = output_embedding_size = self.sent_cell_size * 2
        joint_embedding_size = input_embedding_size + 2

        # Only GRU Model
        self.enc_cell = nn.GRU(
            input_size=joint_embedding_size,
            hidden_size=self.ctx_cell_size,
            num_layers=self.num_layer,
            dropout=0,
            bidirectional=False
        )

        # nn.Linear args --> input size, output size, bias(default true)
        self.attribute_fc1 = nn.Sequential(
            nn.Linear(self.da_embed_size, self.da_hidden_size),
            nn.Tanh()
        )

        cond_embedding_size = self.topic_embed_size + (2 * self.meta_embed_size) + self.ctx_cell_size
        recog_input_size = cond_embedding_size + output_embedding_size
        if self.use_hcf:
            recog_input_size += self.da_embed_size
        self.recog_mulogvar_net = nn.Linear(recog_input_size, self.latent_size * 2)
        self.prior_mulogvar_net = nn.Sequential(
            nn.Linear(cond_embedding_size, np.maximum(self.latent_size * 2, 100)),
            nn.Tanh(),
            nn.Linear(np.maximum(self.latent_size * 2, 100), self.latent_size * 2)
        )

        # BOW Loss Function
        gen_input_size = cond_embedding_size + self.latent_size
        self.bow_project = nn.Sequential(
            nn.Linear(gen_input_size, self.bow_hidden_size),
            nn.Tanh(),
            nn.Dropout(1 - self.keep_prob),
            nn.Linear(self.bow_hidden_size, self.vocab_size)
        )

        # Y Loss Function
        self.da_project = None
        if self.use_hcf:
            self.da_project = nn.Sequential(
                nn.Linear(gen_input_size, self.act_hidden_size),
                nn.Tanh(),
                nn.Dropout(1 - self.keep_prob),
                nn.Linear(self.act_hidden_size, self.da_size)
            )
            dec_input_size = gen_input_size + self.da_embed_size
        else:
            dec_input_size = gen_input_size

        # Decoder
        if self.num_layer > 1:
            self.dec_init_state_net = nn.ModuleList(
                [nn.Linear(dec_input_size, self.dec_cell_size) for _ in range(self.num_layer)]
            )
        else:
            self.dec_init_state_net = nn.Linear(dec_input_size, self.dec_cell_size)
        dec_input_embedding_size = self.embed_size
        if self.use_hcf:
            dec_input_embedding_size += self.da_hidden_size
        self.dec_cell = nn.GRU(
            input_size=dec_input_embedding_size,
            hidden_size=self.dec_cell_size,
            num_layers=self.num_layer,
            dropout=1 - self.keep_prob,
            bidirectional=False
        )
        self.dec_cell_project = nn.Linear(self.dec_cell_size, self.vocab_size)

    def get_converted_info_to_device_context(self, feed_dict):
        context_lens = feed_dict['context_lens'].to(self.device).squeeze(-1)
        input_contexts = feed_dict['vec_context'].to(self.device)
        floors = feed_dict['vec_floors'].to(self.device)
        topics = feed_dict['topics'].to(self.device).squeeze(-1)
        my_profile = feed_dict['my_profile'].to(self.device)
        ot_profile = feed_dict['ot_profile'].to(self.device)

        return context_lens, input_contexts, floors, topics, my_profile, ot_profile

    def get_converted_info_to_device_output(self, feed_dict):
        out_tok = feed_dict['vec_outs'].to(self.device)
        out_das = feed_dict['out_das'].to(self.device).squeeze(-1)
        output_lens = feed_dict['out_lens'].to(self.device).squeeze(-1)

        return out_tok, out_das, output_lens

    def get_encoder_state(self, input_contexts, floors, is_train,
                          context_lens, max_dialog_len, max_seq_len):

        input_contexts = input_contexts.view(-1, max_seq_len)
        input_embedded = self.word_embedding(input_contexts)

        # only use bi-rnn
        if self.sent_type == 'bi-rnn':
            input_embedding, sent_size = get_bi_rnn_encode(
                input_embedded,
                self.bi_sent_cell,
                self.max_tokenized_sent_size
            )
        else:
            raise ValueError("unk sent_type. select one in [bow, rnn, bi-rnn]")

        # reshape input into dialogs
        input_embedding = input_embedding.view(-1, max_dialog_len, sent_size)

        if self.keep_prob < 1.0:
            input_embedding = fnn.dropout(input_embedding, 1 - self.keep_prob, is_train)

        # floors are already converted as one-hot
        floor_one_hot = floors.new_zeros((floors.numel(), 2), dtype=torch.float)
        floor_one_hot.data.scatter_(1, floors.view(-1, 1), 1)
        floor_one_hot = floor_one_hot.view(-1, max_dialog_len, 2)

        joint_embedding = torch.cat([input_embedding, floor_one_hot], 2)

        _, enc_last_state = dynamic_rnn(
            self.enc_cell, joint_embedding,
            sequence_length=context_lens,
            max_len=self.max_tokenized_sent_size
        )

        if self.num_layer > 1:
            enc_last_state = torch.cat([_ for _ in torch.unbind(enc_last_state)], 1)
        else:
            enc_last_state = enc_last_state.squeeze(0)

        return enc_last_state

    def get_sample_from_recog_network(self, local_batch_size, cond_embedding, num_samples, out_das,
                                      output_embedding, is_train_multiple):
        if self.use_hcf:
            attribute_embedding = self.da_embedding(out_das)
            attribute_fc1 = self.attribute_fc1(attribute_embedding)

            ctrl_attribute_embeddings = {
                da: self.da_embedding(torch.ones(local_batch_size, dtype=torch.long, device=self.device) * idx)
                for idx, da in enumerate(self.da_vocab)
            }
            ctrl_attribute_fc1 = {k: self.attribute_fc1(v) for (k, v) in ctrl_attribute_embeddings.items()}

            recog_input = torch.cat([cond_embedding, output_embedding, attribute_fc1], 1)
            ctrl_recog_inputs = {
                k: torch.cat([cond_embedding, output_embedding, v], 1) for (k, v) in ctrl_attribute_fc1.items()
            } if is_train_multiple else {}
        else:
            attribute_embedding = None
            ctrl_attribute_embeddings = None
            recog_input = torch.cat([cond_embedding, output_embedding], 1)
            ctrl_recog_inputs = {
                da: torch.cat([cond_embedding, output_embedding], 1) for idx, da in enumerate(self.da_vocab)
            } if is_train_multiple else {}

        recog_mulogvar = self.recog_mulogvar_net(recog_input)
        recog_mu, recog_logvar = torch.chunk(recog_mulogvar, 2, 1)

        ctrl_latent_samples = {}
        ctrl_recog_mus = {}
        ctrl_recog_logvars = {}
        ctrl_recog_mulogvars = {}

        if is_train_multiple:
            latent_samples = [sample_gaussian(recog_mu, recog_logvar) for _ in range(num_samples)]
            ctrl_recog_mulogvars = {k: self.recog_mulogvar_net(v) for (k, v) in ctrl_recog_inputs.items()}
            for k in ctrl_recog_mulogvars.keys():
                ctrl_recog_mus[k], ctrl_recog_logvars[k] = torch.chunk(ctrl_recog_mulogvars[k], 2, 1)

            ctrl_latent_samples = {
                k: sample_gaussian(ctrl_recog_mus[k], ctrl_recog_logvars[k]) for k in ctrl_recog_mulogvars.keys()
            }
        else:
            latent_samples = [sample_gaussian(recog_mu, recog_logvar)]

        return latent_samples, recog_mu, recog_logvar, recog_mulogvar, ctrl_latent_samples, \
               ctrl_recog_mus, ctrl_recog_logvars, ctrl_recog_mulogvars, attribute_embedding, ctrl_attribute_embeddings
