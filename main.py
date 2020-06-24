import os
import utils
from data_api.cvae_corpus import KGCVAECorpus

CORPUS_CONFIG_PATH = 'config/cvae_corpus_kor.json'
DATASET_CONFIG_PATH = 'config/cvae_dataset_kor.json'
TRAINER_CONFIG_PATH = 'config/cvae_trainer_kor.json'
MODEL_CONFIG_PATH = 'config/cvae_model_kor.json'


def main():
    corpus = KGCVAECorpus(config=utils.load_config(CORPUS_CONFIG_PATH))
    corpus.get_dialog_corpus()


if __name__ == '__main__':
    main()
