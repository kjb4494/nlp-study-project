import torch
import torch.nn.functional as fnn
from model.cvae_static_info import CVAEStaticInfo
from model.model_utils import get_bi_rnn_encode
from model.sample import Sample
from model.decoder import DecodeInputPack, inference_loop, train_loop


class CVAEFeedInfo:
    model_output = None

    def __init__(self, feed_dict, device):
        self.is_train = feed_dict['is_train']
        self.is_train_multiple = feed_dict.get('is_train_multiple', False)
        self.is_test_multi_da = feed_dict.get('is_test_multi_da', True)
        self.num_samples = feed_dict['num_samples']

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

    def get_feed_train(self, s_info: CVAEStaticInfo):
        output_embedded = s_info.word_embedding(self.out_tok)
        if s_info.sent_type == 'bi-rnn':
            output_embedding, _ = get_bi_rnn_encode(
                embedding=output_embedded,
                cell=s_info.bi_sent_cell,
                max_len=s_info.max_tokenized_sent_size
            )
        else:
            raise ValueError('unk sent_type. select one in [bow, rnn, bi-rnn]')

        relation_embedded = s_info.topic_embedding(self.topics)
        enc_last_state = s_info.get_encoder_state(f_info=self)

        cond_list = [relation_embedded, self.my_profile, self.ot_profile, enc_last_state]
        cond_embedding = torch.cat(cond_list, 1)

        sp = Sample(s_info=s_info, f_info=self)
        sp.set_from_recog_network(cond_embedding, output_embedding)
        sp.set_from_prior_network(cond_embedding)

        dip = DecodeInputPack(s_info=s_info, f_info=self)
        dip.set_for_train(
            sp=sp,
            local_batch_size=self.local_batch_size,
            cond_embedding=cond_embedding,
            is_train_multiple=self.is_train_multiple
        )

        dec_outss = []
        ctrl_dec_outs = {}
        # remove eos token
        input_tokens = self.out_tok[:, :-1].clone()
        input_tokens[input_tokens == s_info.eos_id] = 0
        if s_info.dec_keep_prob < 1.0:
            keep_mask = input_tokens.new_empty(input_tokens.size()).bernoulli_(s_info.dec_keep_prob)
            input_tokens = input_tokens * keep_mask
        dec_input_embedded = s_info.word_embedding(input_tokens)
        dec_seq_len = self.output_lens - 1

        dec_input_embedded = fnn.dropout(dec_input_embedded, 1-s_info.keep_prob, True)
        dec_outs, _, final_ctx_state = train_loop(
            cell=s_info.dec_cell,
            output_fn=s_info.dec_cell_project,
            inputs=dec_input_embedded,
            init_state=dip.dec_init_states[0],
            context_vector=sp.attribute_embedding,
            sequence_length=dec_seq_len,
            max_len=s_info.max_tokenized_sent_size - 1
        )

        if self.is_train_multiple:
            for i in range(1, self.num_samples):
                temp_outs, _, _ = inference_loop(
                    cell=s_info.dec_cell,
                    output_fn=s_info.dec_cell_project,
                    embeddings=s_info.word_embedding,
                    encoder_state=dip.dec_init_states[i],
                    start_of_sequence_id=s_info.go_id,
                    end_of_sequence_id=s_info.eos_id,
                    maximum_length=s_info.max_tokenized_sent_size,
                    context_vector=sp.attribute_embedding,
                    decode_type='greedy'
                )
                dec_outss.append(temp_outs)
            for key, value in sp.ctrl_attribute_embeddings.items():
                ctrl_dec_outs[key], _, _ = inference_loop(
                    cell=s_info.dec_cell,
                    output_fn=s_info.dec_cell_project,
                    embeddings=s_info.word_embedding,
                    encoder_state=dip.dec_init_states[key],
                    start_of_sequence_id=s_info.go_id,
                    end_of_sequence_id=s_info.eos_id,
                    maximum_length=s_info.max_tokenized_sent_size,
                    context_vector=value,
                    decode_type='greedy'
                )

        model_output = {
            'dec_out': dec_outs,
            'dec_outss': dec_outss,
            'ctrl_dec_out': ctrl_dec_outs,
            'final_ctx_state': final_ctx_state,
            'bow_logit': dip.bow_logit,
            'da_logit': dip.da_logits[0],
            'out_token': self.out_tok,
            'out_das': self.out_das,
            'recog_mulogvar': sp.recog_mulogvar,
            'prior_mulogvar': sp.prior_mulogvar,
            'context_lens': self.context_lens,
            'vec_context': self.input_contexts
        }
        return model_output

    def get_feed_inference(self, s_info: CVAEStaticInfo):
        relation_embedded = s_info.topic_embedding(self.topics)
        enc_last_state = s_info.get_encoder_state(f_info=self)

        cond_list = [relation_embedded, self.my_profile, self.ot_profile, enc_last_state]
        cond_embedding = torch.cat(cond_list, 1)

        sp = Sample(s_info=s_info, f_info=self)
        sp.set_from_prior_network(cond_embedding)

        dip = DecodeInputPack()
        dip.set_for_test(
            sp=sp,
            local_batch_size=self.local_batch_size,
            cond_embedding=cond_embedding
        )

        sp.ctrl_attribute_embeddings = {
            da: s_info.da_embedding(torch.ones(self.local_batch_size, dtype=torch.long, device=s_info.device) * idx)
            for idx, da in enumerate(s_info.da_vocab)
        }

        dec_outss = []
        ctrl_dec_outs = {}
        dec_outs, _, final_ctx_state = inference_loop(
            cell=s_info.dec_cell,
            output_fn=s_info.dec_cell_project,
            embeddings=s_info.word_embedding,
            encoder_state=dip.dec_init_states[0],
            start_of_sequence_id=s_info.go_id,
            end_of_sequence_id=s_info.eos_id,
            maximum_length=s_info.max_tokenized_sent_size,
            context_vector=dip.pred_attribute_embeddings[0],
            decode_type='greedy'
        )

        for i in range(1, self.num_samples):
            temp_outs, _, _ = inference_loop(
                cell=s_info.dec_cell,
                output_fn=s_info.dec_cell_project,
                embeddings=s_info.word_embedding,
                encoder_state=dip.dec_init_states[i],
                start_of_sequence_id=s_info.go_id,
                end_of_sequence_id=s_info.eos_id,
                maximum_length=s_info.max_tokenized_sent_size,
                context_vector=dip.pred_attribute_embeddings[i],
                decode_type='greedy'
            )
            dec_outss.append(temp_outs)
        if self.is_test_multi_da:
            for key, value in sp.ctrl_attribute_embeddings.items():
                ctrl_dec_outs[key], _, _ = inference_loop(
                    cell=s_info.dec_cell,
                    output_fn=s_info.dec_cell_project,
                    embeddings=s_info.word_embedding,
                    encoder_state=dip.dec_init_states[0],
                    start_of_sequence_id=s_info.go_id,
                    end_of_sequence_id=s_info.eos_id,
                    maximum_length=s_info.max_tokenized_sent_size,
                    context_vector=value,
                    decode_type='greedy'
                )

        model_output = {
            'dec_out': dec_outs, 
            'dec_outss': dec_outss, 
            'ctrl_dec_out': ctrl_dec_outs,
            'final_ctx_state': final_ctx_state,
            'bow_logit': dip.bow_logit,
            'da_logit': dip.da_logits[0],
            'prior_mulogvar': sp.prior_mulogvar,
            'context_lens': self.context_lens,
            'vec_context': self.input_contexts
        }
        return model_output
