
import torch.nn as nn
from model.cvae_static_info import CVAEStaticInfo
from model.cvae_feed_info import CVAEFeedInfo
from model.index2sent import SentPack, index2sent


class CVAEModel(nn.Module):
    def __init__(self, model_config, vocab_class):
        super(CVAEModel, self).__init__()
        self.s_info = CVAEStaticInfo(model_config=model_config, vocab_class=vocab_class)

        # Tensor Value For Parameters() in Optimizer
        self.topic_embedding = self.s_info.topic_embedding
        self.da_embedding = self.s_info.da_embedding
        self.word_embedding = self.s_info.word_embedding
        self.bi_sent_cell = self.s_info.bi_sent_cell
        self.enc_cell = self.s_info.enc_cell
        self.attribute_fc1 = self.s_info.attribute_fc1
        self.recog_mulogvar_net = self.s_info.recog_mulogvar_net
        self.prior_mulogvar_net = self.s_info.prior_mulogvar_net
        self.bow_project = self.s_info.bow_project
        self.da_project = self.s_info.da_project
        self.dec_init_state_net = self.s_info.dec_init_state_net
        self.dec_cell = self.s_info.dec_cell
        self.dec_cell_project = self.s_info.dec_cell_project

    # arg: torch DataLoader
    def forward(self, feed_dict):
        device = self.s_info.device
        f_info = CVAEFeedInfo(feed_dict, device)
        if f_info.is_train:
            model_output: dict = f_info.get_feed_train(s_info=self.s_info)
        else:
            model_output: dict = f_info.get_feed_inference(s_info=self.s_info)
            model_output.update({
                'out_token': f_info.out_tok,
                'out_das': f_info.out_das
            })

        sent_pack: SentPack = index2sent(
            input_contexts=f_info.input_contexts,
            context_lens=f_info.context_lens,
            model_output=model_output,
            feed_real=True,
            eos_id=self.s_info.eos_id,
            vocab=self.s_info.vocab,
            da_vocab=self.s_info.da_vocab
        )

        model_output.update({
            'output_sents': sent_pack.output_sents,
            'ctrl_output_sents': sent_pack.ctrl_output_sents,
            'sampled_output_sents': sent_pack.sampled_output_sents,
            'output_das': sent_pack.output_logits,
            'real_output_sents': sent_pack.real_output_sents,
            'real_output_das': sent_pack.real_output_logits,
            'context_sents': sent_pack.input_context_sents
        })

        return model_output
