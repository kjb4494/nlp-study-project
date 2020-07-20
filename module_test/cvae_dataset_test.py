import utils
from data_api.dataset import CVAEDataset
from module_test.cvae_corpus_test import get_corpus


def test_main():
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


if __name__ == '__main__':
    CORPUS_CONFIG_PATH = '../config_for_test_modules/cvae_corpus_kor.json'
    DATASET_CONFIG_PATH = '../config_for_test_modules/cvae_dataset_kor.json'

    corpus = get_corpus(CORPUS_CONFIG_PATH)

    test_main()
