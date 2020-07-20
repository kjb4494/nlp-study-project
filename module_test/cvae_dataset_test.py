import utils
from torch.utils.data import DataLoader
from data_api.dataset import CVAEDataset
from module_test.cvae_corpus_test import get_corpus
from data_api.data_loader import get_cvae_collate


def test_main():
    corpus = get_corpus(CORPUS_CONFIG_PATH)
    train_dial = corpus.get_dialog_corpus().get('train')
    train_meta = corpus.get_meta_corpus().get('train')
    dataset_config = utils.load_config(DATASET_CONFIG_PATH)
    train_set = CVAEDataset(
        name='train',
        data=train_dial,
        meta_data=train_meta,
        config=dataset_config
    )
    print('data_lens: ', train_set.data_lens)
    print('indexes: ', train_set.indexes)
    # data 부분
    for index, item in enumerate(train_set.data):
        print('===== ', index, '번 데이터 =====')
        for key, value in item.items():
            print(key, ': ', value)
        if index >= 50:
            break


def get_dataset(corpus_config_path, dataset_config_path):
    corpus = get_corpus(corpus_config_path)
    dialog_corpus = corpus.get_dialog_corpus()
    meta_corpus = corpus.get_meta_corpus()
    dataset_config = utils.load_config(dataset_config_path)
    train_set = CVAEDataset(
        name='train',
        data=dialog_corpus.get('train'),
        meta_data=meta_corpus.get('train'),
        config=dataset_config
    )
    test_set = CVAEDataset(
        name='test',
        data=dialog_corpus.get('test'),
        meta_data=meta_corpus.get('test'),
        config=dataset_config
    )
    valid_set = CVAEDataset(
        name='valid',
        data=dialog_corpus.get('valid'),
        meta_data=meta_corpus.get('valid'),
        config=dataset_config
    )
    return train_set, test_set, valid_set


def get_data_loader(corpus_config_path, dataset_config_path):
    dataset_config = utils.load_config(dataset_config_path)
    cvae_collate = get_cvae_collate(
        tokenized_sent_per_case=dataset_config['tokenized_sent_per_case'],
        max_tokenized_sent_size=dataset_config['max_tokenized_sent_size']
    )
    train_set, test_set, valid_set = get_dataset(corpus_config_path, dataset_config_path)
    train_loader = DataLoader(train_set, batch_size=100, shuffle=True, collate_fn=cvae_collate)
    test_loader = DataLoader(test_set, batch_size=100, shuffle=False, collate_fn=cvae_collate)
    valid_loader = DataLoader(valid_set, batch_size=100, shuffle=False, collate_fn=cvae_collate)
    return train_loader, test_loader, valid_loader


if __name__ == '__main__':
    CORPUS_CONFIG_PATH = '../config_for_test_modules/cvae_corpus_kor.json'
    DATASET_CONFIG_PATH = '../config_for_test_modules/cvae_dataset_kor.json'
    test_main()
