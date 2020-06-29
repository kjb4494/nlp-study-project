import os
import utils
from data_api.cvae_corpus import KGCVAECorpus
from data_api.dataset import CVAEDataset

CORPUS_CONFIG_PATH = 'config/cvae_corpus_kor.json'
DATASET_CONFIG_PATH = 'config/cvae_dataset_kor.json'
TRAINER_CONFIG_PATH = 'config/cvae_trainer_kor.json'
MODEL_CONFIG_PATH = 'config/cvae_model_kor.json'


def main():
    corpus = KGCVAECorpus(config=utils.load_config(CORPUS_CONFIG_PATH))
    dial_corpus = corpus.get_dialog_corpus()
    meta_corpus = corpus.get_meta_corpus()

    train_dial = dial_corpus.get('train')
    test_dial = dial_corpus.get('test')
    valid_dial = dial_corpus.get('valid')

    train_meta = meta_corpus.get('train')
    test_meta = meta_corpus.get('test')
    valid_meta = meta_corpus.get('valid')

    dataset_config = utils.load_config(DATASET_CONFIG_PATH)
    utt_per_case = dataset_config['utt_per_case']
    max_utt_size = dataset_config['max_utt_len']

    CVAEDataset('train', train_dial, train_meta, dataset_config)


if __name__ == '__main__':
    main()
