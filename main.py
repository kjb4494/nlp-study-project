import os
import utils
from torch.utils.data import DataLoader
from data_api.cvae_corpus import KGCVAECorpus
from data_api.dataset import CVAEDataset
from data_api.data_loader import get_cvae_collate
from model.cvae import CVAEModel
from trainer.trainer import CVAETrainer


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

    train_set = CVAEDataset('train', train_dial, train_meta, dataset_config)
    test_set = CVAEDataset('test', test_dial, test_meta, dataset_config)
    valid_set = CVAEDataset('valid', valid_dial, valid_meta, dataset_config)

    cvae_collate = get_cvae_collate(
        tokenized_sent_per_case=dataset_config['tokenized_sent_per_case'],
        max_tokenized_sent_size=dataset_config['max_tokenized_sent_size']
    )

    train_loader = DataLoader(train_set, batch_size=100, shuffle=True, collate_fn=cvae_collate)
    test_loader = DataLoader(test_set, batch_size=100, shuffle=False, collate_fn=cvae_collate)
    valid_loader = DataLoader(valid_set, batch_size=100, shuffle=False, collate_fn=cvae_collate)

    print(vars(train_loader))

    trainer_config = utils.load_config(TRAINER_CONFIG_PATH)
    model_config = utils.load_config(MODEL_CONFIG_PATH)

    target_model = CVAEModel(model_config=model_config, vocab_class=corpus)
    target_model.cpu()
    cvae_trainer = CVAETrainer(trainer_config, target_model)
    output_reports = cvae_trainer.experiment(train_loader, valid_loader, test_loader)


if __name__ == '__main__':
    main()
