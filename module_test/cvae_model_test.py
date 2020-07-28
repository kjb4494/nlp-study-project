import utils
from model.cvae import CVAEModel
from module_test.cvae_corpus_test import get_corpus


def kg_cvae_model():
    cvae_model = CVAEModel(
        model_config=utils.load_config(MODEL_CONFIG_PATH),
        vocab_class=get_corpus(corpus_config_path=CORPUS_CONFIG_PATH)
    )
    topic_embedding = cvae_model.topic_embedding



def test_main():
    kg_cvae_model()


if __name__ == '__main__':
    CORPUS_CONFIG_PATH = '../config_for_test_modules/cvae_corpus_kor.json'
    MODEL_CONFIG_PATH = '../config_for_test_modules/cvae_model_kor.json'
    test_main()
