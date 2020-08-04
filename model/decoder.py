from model.sample import Sample
from model.cvae_static_info import CVAEStaticInfo
from .model_utils import dynamic_rnn
import torch


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
