
import torch
import torch.nn as nn
import torch.nn.functional as fnn

import numpy as np


class CVAEModel(nn.Module):
    def __init__(self, data_config, model_config, vocab_class):
        super(CVAEModel, self).__init__()
        self.data_config = data_config
        self.model_config = model_config
        self.vocab_config = vocab_class

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
                dropout=1-self.keep_prob,
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
