import utils
from torch.utils.data import DataLoader
from data_api.data_loader import get_cvae_collate
from module_test.cvae_dataset_test import get_dataset


def test_main():
    dataset_config = utils.load_config(DATASET_CONFIG_PATH)
    train_set, test_set, valid_set = get_dataset(CORPUS_CONFIG_PATH, DATASET_CONFIG_PATH)
    cvae_collate = get_cvae_collate(
        tokenized_sent_per_case=dataset_config['tokenized_sent_per_case'],
        max_tokenized_sent_size=dataset_config['max_tokenized_sent_size']
    )

    train_loader = DataLoader(train_set, batch_size=100, shuffle=True, collate_fn=cvae_collate)

    # Without collate function
    # train_loader = DataLoader(train_set, batch_size=100, shuffle=True)


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
