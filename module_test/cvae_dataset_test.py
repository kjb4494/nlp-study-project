import utils
from data_api.dataset import CVAEDataset
from module_test.cvae_corpus_test import get_corpus


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


if __name__ == '__main__':
    CORPUS_CONFIG_PATH = '../config_for_test_modules/cvae_corpus_kor.json'
    DATASET_CONFIG_PATH = '../config_for_test_modules/cvae_dataset_kor.json'
    test_main()
