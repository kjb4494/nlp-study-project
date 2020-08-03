from model.sample import Sample
from model.cvae_static_info import CVAEStaticInfo


class DecodeInputPack:
    # Common
    da_logits = None
    bow_logit = None
    dec_inputs = None
    dec_init_states = None

    # For Train
    ctrl_dec_inputs = None
    ctrl_dec_states = None

    # For Test
    pred_attribute_embeddings = None

    def set_for_train(self, sp: Sample, local_batch_size, cond_embedding, is_train_multiple):
        pass

    def set_for_test(self):
        pass


def inference_loop(cell, output_fn, embeddings, encoder_state, start_of_sequence_id,
                   end_of_sequence_id, maximum_length, num_decoder_symbols, context_vecotr, decode_type='greedy'):
    pass


def train_loop(cell, output_fn, inputs, init_state, context_vector, sequence_length, max_len):
    pass
