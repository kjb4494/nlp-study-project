import utils
from data_api.cvae_corpus import KGCVAECorpus
from pprint import pprint as prt


def test_main():
    print('===== Input Data =====')
    prt(utils.load_config(CORPUS_CONFIG_PATH))
    corpus = KGCVAECorpus(config=utils.load_config(CORPUS_CONFIG_PATH))

    print('===== CORPUS Object =====')
    print('vocab: ', corpus.vocab)
    print('rev_vocab: ', corpus.rev_vocab)
    print('unk_id: ', corpus.unk_id)
    print('topic_vocab: ', corpus.topic_vocab)
    print('rev_topic_vocab: ', corpus.rev_topic_vocab)
    print('dialog_act_vocab: ', corpus.dialog_act_vocab)
    print('rev_dialog_act_vocab: ', corpus.rev_dialog_act_vocab)
    print('word2vec: ', corpus.word2vec)

    print('===== Dialog Corpus =====')
    print(corpus.get_dialog_corpus())

    print('===== Meta Corpus =====')
    print(corpus.get_meta_corpus())


def get_corpus(corpus_config_path):
    return KGCVAECorpus(config=utils.load_config(corpus_config_path))


if __name__ == '__main__':
    CORPUS_CONFIG_PATH = '../config_for_test_modules/cvae_corpus_kor.json'
    test_main()
