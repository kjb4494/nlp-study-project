import utils
from model.cvae import CVAEModel
from module_test.cvae_corpus_test import get_corpus


def kg_cvae_model():
    cvae_model = CVAEModel(
        model_config=utils.load_config(MODEL_CONFIG_PATH),
        vocab_class=get_corpus(corpus_config_path=CORPUS_CONFIG_PATH)
    )
    topic_embedding = cvae_model.topic_embedding
    da_embedding = cvae_model.da_embedding
    word_embedding = cvae_model.word_embedding
    bi_sent_cell = cvae_model.bi_sent_cell
    enc_cell = cvae_model.enc_cell
    attribute_fc1 = cvae_model.attribute_fc1
    recog_mulogvar_net = cvae_model.recog_mulogvar_net
    prior_mulogvar_net = cvae_model.prior_mulogvar_net
    bow_project = cvae_model.bow_project
    da_project = cvae_model.da_project
    dec_init_state_net = cvae_model.dec_init_state_net
    dec_cell = cvae_model.dec_cell
    dec_cell_project = cvae_model.dec_cell_project

    print('Debug stop position')

    cvae_model.cpu()


def test_main():
    kg_cvae_model()


if __name__ == '__main__':
    CORPUS_CONFIG_PATH = '../config_for_test_modules/cvae_corpus_kor.json'
    MODEL_CONFIG_PATH = '../config_for_test_modules/cvae_model_kor.json'
    test_main()
