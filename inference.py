import utils
import torch
import tqdm
from data_api.cvae_corpus import KGCVAECorpus
from data_api.data_loader import get_cvae_collate
from data_api.dataset import CVAEDataset
from torch.utils.data import DataLoader
from model.cvae import CVAEModel
from model.cvae_feed_info import CVAEFeedInfo
from model.index2sent import SentPack, index2sent

CORPUS_CONFIG_PATH = 'config/cvae_corpus_kor.json'
DATASET_CONFIG_PATH = 'config/cvae_dataset_kor.json'
TRAINER_CONFIG_PATH = 'config/cvae_trainer_kor.json'
MODEL_CONFIG_PATH = 'config/cvae_model_kor.json'
MODEL_PATH = 'output/model/kor_cvae_model_45.pth'


def inference(model: CVAEModel, data_loader, trainer_config):
    outputs = []
    iterator = tqdm.tqdm(data_loader, desc='Inference')
    for model_input in iterator:
        model_input.update({
            'is_train': False,
            'is_test_multi_da': trainer_config['is_test_multi_da'],
            'num_samples': trainer_config['num_samples']
        })
        with torch.no_grad():
            f_info = CVAEFeedInfo(feed_dict=model_input, device=model.s_info.device)
            model_output = f_info.get_feed_inference(s_info=model.s_info)

        sp: SentPack = index2sent(
            input_contexts=model_output['vec_context'],
            context_lens=model_output['context_lens'],
            model_output=model_output,
            feed_real=False,
            eos_id=model.s_info.eos_id,
            vocab=model.s_info.vocab,
            da_vocab=model.s_info.da_vocab
        )

        model_output.update({
            'output_sents': sp.output_sents,
            'ctrl_output_sents': sp.ctrl_output_sents,
            'sampled_output_sents': sp.sampled_output_sents,
            'output_das': sp.output_logits,
            'context_sents': sp.input_context_sents
        })

        output = {'model_input': model_input, 'model_output': model_output}
        outputs.append(output)

    topics = ['LOVER', 'FRIEND']
    profiles = ['STUDENT', 'COLLEGIAN', 'CIVILIAN', 'FEMALE', 'MALE']
    profiles_len = len(profiles)

    results = []

    for output in outputs:
        segment_indices = []
        context_lens = output['model_output']['context_lens'].tolist()
        for i in range(1, len(context_lens)):
            if context_lens[i] <= context_lens[i - 1]:
                segment_indices.append(i - 1)
        segment_indices.append(len(context_lens) - 1)
        for i in segment_indices:
            out_dict = {
                'relation': topics[output['model_input']['topics'][i]],
                'my_profile': [profiles[j] for j in range(profiles_len) if output['model_input']['my_profile'][i][j]],
                'ot_profile': [profiles[j] for j in range(profiles_len) if output['model_input']['ot_profile'][i][j]],
                'contexts': output['model_output']['context_sents'][i],
                'generated': output['model_output']['output_sents'][i],
                'samples': [output['model_output']['sampled_output_sents'][j][i]
                            for j in range(trainer_config['num_samples'] - 1)]
            }
            if trainer_config["is_test_multi_da"]:
                for da in model.s_info.da_vocab:
                    out_dict[da] = output['model_output']['ctrl_output_sents'][da][i]
            out_dict['predicted_sentiment'] = output['model_output']['output_das'][i]
            results.append(out_dict)

    return results


def main_code():
    corpus_config = utils.load_config(CORPUS_CONFIG_PATH)
    corpus_config['load_vocab'] = True

    corpus = KGCVAECorpus(corpus_config)
    dial_corpus = corpus.get_dialog_corpus()
    meta_corpus = corpus.get_meta_corpus()

    test_dial = dial_corpus.get('test')
    test_meta = meta_corpus.get('test')

    dataset_config = utils.load_config(DATASET_CONFIG_PATH)

    test_set = CVAEDataset('test', test_dial, test_meta, dataset_config)
    cvae_collate = get_cvae_collate(
        tokenized_sent_per_case=dataset_config['tokenized_sent_per_case'],
        max_tokenized_sent_size=dataset_config['max_tokenized_sent_size']
    )
    test_loader = DataLoader(test_set, batch_size=100, shuffle=False, collate_fn=cvae_collate)

    trainer_config = utils.load_config(TRAINER_CONFIG_PATH)
    model_config = utils.load_config(MODEL_CONFIG_PATH)

    model = CVAEModel(model_config=model_config, vocab_class=corpus)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    model.cpu()

    result = inference(model, test_loader, trainer_config)
    print(result)


if __name__ == '__main__':
    main_code()
