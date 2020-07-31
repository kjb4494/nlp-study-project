import torch.nn.functional as fnn
import torch
from model.cvae_static_info import CVAEStaticInfo
from model.model_utils import get_bi_rnn_encode, dynamic_rnn


class CVAEFeedInfo:
    model_output = None

    def __init__(self, info: CVAEStaticInfo, feed_dict):
        self.info = info

        self.is_train = feed_dict['is_train']
        self.is_train_multiple = feed_dict.get('is_train_multiple', False)
        self.is_test_multi_da = feed_dict.get('is_test_multi_da', True)
        self.num_samples = feed_dict['num_samples']

        device = info.device

        # Device Context Info
        self.context_lens = feed_dict['context_lens'].to(device).squeeze(-1)
        self.input_contexts = feed_dict['vec_context'].to(device)
        self.floors = feed_dict['vec_floors'].to(device)
        self.topics = feed_dict['topics'].to(device).squeeze(-1)
        self.my_profile = feed_dict['my_profile'].to(device)
        self.ot_profile = feed_dict['ot_profile'].to(device)

        # Device Output Info
        self.out_tok = feed_dict['vec_outs'].to(device)
        self.out_das = feed_dict['out_das'].to(device).squeeze(-1)
        self.output_lens = feed_dict['out_lens'].to(device).squeeze(-1)

        # Init Common Value Settings
        self.local_batch_size = self.input_contexts.size(0)
        self.max_dialog_len = self.input_contexts.size(1)
        self.max_seq_len = self.input_contexts.size(-1)

        if self.is_train:
            self._feed_train()
        else:
            self._feed_inference()

    def _feed_train(self):
        output_embedded = self.info.word_embedding(self.out_tok)
        if self.info.sent_type == 'bi-rnn':
            output_embedding, _ = get_bi_rnn_encode(
                embedding=output_embedded,
                cell=self.info.bi_sent_cell,
                max_len=self.info.max_tokenized_sent_size
            )
        else:
            raise ValueError('unk sent_type. select one in [bow, rnn, bi-rnn]')

        relation_embedded = self.info.topic_embedding(self.topics)
        enc_last_state = self._get_encoder_state()

        cond_list = [relation_embedded, self.my_profile, self.ot_profile, enc_last_state]
        cond_embedding = torch.cat(cond_list, 1)

        sample_result = self._get_sample_from_recog_network(
            cond_embedding=cond_embedding,
            output_embedding=output_embedding
        )
        # Sample from prior network
        prior_mulogvar = self.info.prior_mulogvar_net(cond_embedding)
        prior_mu, prior_logvar = torch.chunk(prior_mulogvar, 2, 1)

        ctrl_gen_inputs = None
        if self.is_train_multiple:
            ctrl_gen_inputs = {
                k: torch.cat([cond_embedding, v], 1)
                for k, v in ctrl_latent_samples.items()
            }

        # Decoder test input
        gen_inputs = [torch.cat([cond_embedding, latent_sample], 1) for latent_sample in latent_samples]
        bow_logit = self.info.bow_project(gen_inputs[0])

        if self.info.use_hcf:
            da_logits = [self.info.da_project(gen_input) for gen_input in gen_inputs]
            da_probs = [fnn.softmax(da_logit, dim=1) for da_logit in da_logits]
            pred_attribute_embeddings = [torch.matmul(da_prob, self.info.da_embedding.weight) for da_prob in da_probs]
            selected_attr_embedding = pred_attribute_embeddings
            dec_inputs = [torch.cat((gen_input, selected_attr_embedding[i]), 1) for i, gen_input in enumerate(gen_inputs)]
        else:
            da_logits = [gen_input.new_zeros(self.local_batch_size, self.info.da_size) for gen_input in gen_inputs]
            dec_inputs = gen_inputs
            pred_attribute_embeddings = []

        # decoder
        if self.info.num_layer > 1:
            dec_init_states = [
                [self.info.dec_init_state_net[i](dec_input) for i in range(self.info.num_layer)]
                for dec_input in dec_inputs
            ]
            dec_init_states = [torch.stack(dec_init_state) for dec_init_state in dec_init_states]
        else:
            dec_init_states = [self.info.dec_init_state_net(dec_input).unsqueeze(0) for dec_input in dec_inputs]

    def _feed_inference(self):
        pass

    def _get_encoder_state(self):
        input_contexts = self.input_contexts.view(-1, self.max_seq_len)
        relation_embedded = self.info.topic_embedding(self.topics)
        input_embedded = self.info.word_embedding(input_contexts)

        if self.info.sent_type == 'bi-rnn':
            input_embedding, sent_size = get_bi_rnn_encode(
                embedding=input_embedded,
                cell=self.info.bi_sent_cell,
                max_len=self.info.max_tokenized_sent_size
            )
        else:
            raise ValueError("unk sent_type. select one in [bow, rnn, bi-rnn]")

        input_embedding = input_embedding.view(-1, self.max_dialog_len, sent_size)

        if self.info.keep_prob < 1.0:
            input_embedding = fnn.dropout(input_embedding, 1 - self.info.keep_prob, self.is_train)

        floor_one_hot = self.floors.new_zeros((self.floors.numel(), 2), dtype=torch.float)
        floor_one_hot.data.scatter_(1, self.floors.view(-1, 1), 1)
        floor_one_hot = floor_one_hot.view(-1, self.max_dialog_len, 2)

        joint_embedding = torch.cat([input_embedding, floor_one_hot], 2)
        # 미완성 함수
        _, enc_last_state = dynamic_rnn(
            cell=self.info.enc_cell,
            inputs=joint_embedding,
            sequence_length=self.context_lens,
            max_len=self.info.max_tokenized_sent_size
        )

        if self.info.num_layer > 1:
            enc_last_state = torch.cat([_ for _ in torch.unbind(enc_last_state)], 1)
        else:
            enc_last_state = enc_last_state.squeeze(0)

        return enc_last_state
