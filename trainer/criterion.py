import torch
import torch.nn as nn
import torch.nn.functional as fnn


class CVAELoss(nn.Module):
    def __init__(self, config):
        super(CVAELoss, self).__init__()
        self.use_hcf = config['use_hcf']
        self.full_kl_step = config['full_kl_step']

    # 상속
    def forward(self, model_output, model_input, current_step, is_train, is_valid):
        out_token = model_output["out_token"]
        out_das = model_output["out_das"]
        da_logit = model_output["da_logit"]
        bow_logit = model_output["bow_logit"]
        dec_out = model_output["dec_out"]

        # Calculate seq_loss
        labels = out_token.clone()[:, 1:]
        label_mask = torch.sign(labels).detach().float()
        dec_out = dec_out.contiguous()
        rc_loss = fnn.cross_entropy(
            dec_out.view(-1, dec_out.size(-1)),
            labels.reshape(-1),
            reduction='none'
        ).view(dec_out.size()[:-1])
        rc_loss = torch.sum(rc_loss * label_mask, 1)
        avg_rc_loss = torch.mean(rc_loss)
        rc_ppl = torch.exp(torch.sum(rc_loss) / torch.sum(label_mask))
        bow_loss = -fnn.log_softmax(bow_logit, dim=1).gather(1, labels) * label_mask
        bow_loss = torch.sum(bow_loss, 1)
        avg_bow_loss = torch.mean(bow_loss)

        if self.use_hcf:
            avg_sentiment_loss = fnn.cross_entropy(da_logit, out_das)
            _, da_logits = torch.max(da_logit, 1)
            avg_sentiment_acc = (da_logits == out_das).float().sum() / out_das.shape[0]
        else:
            avg_sentiment_loss = avg_bow_loss.new_tensor(0)
            avg_sentiment_acc = 0.0

        # Calculate loss
        losses = {
            'avg_rc_loss': avg_rc_loss,
            'rc_ppl': rc_ppl,
            'avg_sentiment_loss': avg_sentiment_loss,
            'avg_sentiment_acc': avg_sentiment_acc,
            'avg_bow_loss': avg_bow_loss
        }

        if is_train:
            recog_mulogvar = model_output['recog_mulogvar']
            prior_mulogvar = model_output['prior_mulogvar']
            recog_mu, recog_logvar = torch.chunk(recog_mulogvar, 2, 1)
            prior_mu, prior_logvar = torch.chunk(prior_mulogvar, 2, 1)

            # Calculate KL loss
            def gaussian_kld():
                return -0.5 * torch.sum(
                    1 + (recog_logvar - prior_logvar) -
                    torch.div(torch.pow(prior_mu - recog_mu, 2), torch.exp(prior_logvar)) -
                    torch.div(torch.exp(recog_logvar), torch.exp(prior_logvar)),
                    1
                )

            avg_kld = torch.mean(gaussian_kld())
            kl_weights = min((current_step / self.full_kl_step), 1.0) if not is_valid else 1.0
            losses.update({
                'avg_kl_loss': avg_kld,
                'kl_weights': kl_weights
            })
            elbo = avg_rc_loss + kl_weights * avg_kld
            aug_elbo = avg_bow_loss + avg_sentiment_loss + elbo
        else:
            elbo = avg_rc_loss
            aug_elbo = avg_bow_loss + avg_sentiment_loss + elbo
        losses['main_loss'] = aug_elbo
        return losses
