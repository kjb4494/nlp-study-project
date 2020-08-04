import torch
import torch.nn.functional as fnn
from model.sample import Sample
from model.cvae_static_info import CVAEStaticInfo
from model.cvae_feed_info import CVAEFeedInfo
from .model_utils import dynamic_rnn


class DecodeInputPack:
    # Common
    da_logits = None
    bow_logit = None
    dec_inputs = None
    dec_init_states = None

    # For Train
    ctrl_dec_inputs = None
    ctrl_dec_init_states = None

    # For Test
    pred_attribute_embeddings = None

    def __init__(self, s_info: CVAEStaticInfo, f_info: CVAEFeedInfo):
        self.s_info = s_info
        self.f_info = f_info

    def set_for_train(self, sp: Sample, local_batch_size, cond_embedding, is_train_multiple):
        gen_inputs = [torch.cat([cond_embedding, latent_sample], 1) for latent_sample in sp.latent_samples]
        bow_logit = self.s_info.bow_project(gen_inputs[0])
        ctrl_dec_inputs = {}
        ctrl_dec_init_states = {}

        if self.s_info.use_hcf:
            da_logits = [self.s_info.da_project(gen_input) for gen_input in gen_inputs]
            da_probs = [fnn.softmax(da_logit, dim=1) for da_logit in da_logits]

            selected_attr_embedding = sp.attribute_embedding
            dec_inputs = [torch.cat((gen_input, selected_attr_embedding), 1) for gen_input in gen_inputs]

            if self.f_info.is_train_multiple:
                ctrl_gen_inputs = {
                    k: torch.cat([cond_embedding, v], 1) for k, v in sp.ctrl_latent_samples.items()
                }
                ctrl_dec_inputs = {
                    k: torch.cat((ctrl_gen_inputs[k], sp.ctrl_attribute_embeddings[k]), 1)
                    for k in ctrl_gen_inputs.keys()
                }
        else:
            da_logits = [gen_input.new_zeros(local_batch_size, self.s_info.da_size)
                         for gen_input in gen_inputs]
            dec_inputs = gen_inputs

        # decoder
        if self.s_info.num_layer > 1:
            dec_init_states = [
                [self.s_info.dec_init_state_net[i](dec_input) for i in range(self.s_info.num_layer)]
                for dec_input in dec_inputs
            ]
            dec_init_states = [torch.stack(dec_init_state) for dec_init_state in dec_init_states]
            if self.f_info.is_train_multiple:
                for k, v in ctrl_dec_inputs.items():
                    ctrl_dec_init_states[k] = [
                        self.s_info.dec_init_state_net[i](v) for i in range(self.s_info.num_layer)
                    ]
        else:
            dec_init_states = [self.s_info.dec_init_state_net(dec_input).unsqueeze(0) for dec_input in dec_inputs]
            if self.f_info.is_train_multiple:
                ctrl_dec_init_states = {
                    k: self.s_info.dec_init_state_net(v).unsqueeze(0) for k, v in ctrl_dec_inputs.items()
                }

        # Setting results
        self.da_logits = da_logits
        self.bow_logit = bow_logit
        self.dec_inputs = dec_inputs
        self.dec_init_states = dec_init_states
        self.ctrl_dec_inputs = ctrl_dec_inputs
        self.ctrl_dec_init_states = ctrl_dec_init_states

    def set_for_test(self, sp: Sample, local_batch_size, cond_embedding):
        # _get_dec_input_test
        pass


def inference_loop(cell, output_fn, embeddings, encoder_state, start_of_sequence_id,
                   end_of_sequence_id, maximum_length, context_vector, decode_type='greedy'):
    batch_size = encoder_state.size(1)
    outputs = []
    context_state = []
    cell_state = encoder_state
    cell_output = None

    for time in range(maximum_length):
        if cell_output is None:
            next_input_id = encoder_state.new_full((batch_size,), start_of_sequence_id, dtype=torch.long)
            done = encoder_state.new_zeros(batch_size, dtype=torch.uint8)
            cell_state = encoder_state
        else:
            cell_output = output_fn(cell_output)
            outputs.append(cell_output)

            if decode_type == 'sample':
                matrix_u = -1.0 * torch.log(-1.0 * torch.log(cell_output.new_empty(cell_output.size()).uniform_()))
                next_input_id = torch.max(cell_output - matrix_u, 1)[1]
            elif decode_type == 'greedy':
                next_input_id = torch.max(cell_output, 1)[1]
            else:
                raise ValueError('unknown decode type')
            next_input_id = next_input_id * (~done).long()  # ???
            done = (next_input_id == end_of_sequence_id) | done
            context_state.append(next_input_id)

        next_input = embeddings(next_input_id)
        if context_vector is not None:
            next_input = torch.cat([next_input, context_vector], 1)
        if done.long().sum() == batch_size:
            break

        cell_output, cell_state = cell(next_input.unsqueeze(1), cell_state)
        # Squeeze the time dimension
        cell_output = cell_output.squeeze(1)

        # zero out done sequences
        cell_output = cell_output * (~done).float().unsqueeze(1)

    dec_out = torch.cat([_.unsqueeze(1) for _ in outputs], 1)
    pad_layer = torch.nn.ConstantPad1d((0, maximum_length - dec_out.shape[1] - 1), 0.0)
    dec_out = dec_out.permute(0, 2, 1)

    dec_out = pad_layer(dec_out)
    dec_out = dec_out.permute(0, 2, 1)
    dec_out[:, dec_out.shape[1]:maximum_length - 1, 0] = 1.0

    out_result = dec_out, cell_state, torch.cat(
        [_.unsqueeze(1) for _ in context_state], 1)

    return out_result


def train_loop(cell, output_fn, inputs, init_state, context_vector, sequence_length, max_len):
    if context_vector is not None:
        inputs = torch.cat(
            [inputs, context_vector.unsqueeze(1).expand(inputs.size(0), inputs.size(1), context_vector.size(1))], 2
        )
    return dynamic_rnn(cell, inputs, sequence_length, max_len, init_state, output_fn) + (None,)
