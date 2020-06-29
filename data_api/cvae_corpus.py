import os
import json
import pickle
import numpy as np
from collections import Counter
from gensim.models.wrappers import FastText
from pprint import pprint


class KGCVAECorpus:
    reserved_token_for_gen = ['<pad>', '<unk>', '<sos>', '<eos>']
    reserved_token_for_dialog = ['<s>', '<d>', '</s>']

    # corpus 리스크의 정보 위치
    dialog_id = 0
    meta_id = 1
    utt_id = 2

    vocab = None
    rev_vocab = None
    unk_id = None
    topic_vocab = None
    rev_topic_vocab = None
    dialog_act_vocab = None
    rev_dialog_act_vocab = None
    word2vec = None

    def __init__(self, config):
        dir_path = config['data_dir']

        train_file_name = config['train_filename']
        test_file_name = config['test_filename']
        valid_file_name = config['valid_filename']

        train_file_path = os.path.join(dir_path, train_file_name)
        test_file_path = os.path.join(dir_path, test_file_name)
        valid_file_path = os.path.join(dir_path, valid_file_name)

        with open(train_file_path, 'r', encoding='utf-8') as reader:
            train_corpus_data = json.load(reader)

        with open(test_file_path, 'r', encoding='utf-8') as reader:
            test_corpus_data = json.load(reader)

        with open(valid_file_path, 'r', encoding='utf-8') as reader:
            valid_corpus_data = json.load(reader)

        self.train_corpus = self._process(train_corpus_data)
        self.test_corpus = self._process(test_corpus_data)
        self.valid_corpus = self._process(valid_corpus_data)

        exists_load_vocab = config.get('load_vocab', False)
        if exists_load_vocab:
            self._load_vocab(config['vocab_path'])
        else:
            self._build_vocab(config['max_vocab_count'])
            self._save_vocab(config['vocab_path'])

        self._load_word2vec(config['word2vec_path'], config['word2vec_path'])

    @staticmethod
    def _process(json_data):
        new_dialog = []
        new_meta = []
        new_utts = []
        bod_utts = ['<s>', '<d>', '</s>']
        all_lenes = []
        for session_data in json_data:
            session_utts = session_data['utts']
            a_info = session_data['A']
            b_info = session_data['B']
            lower_utts = [(caller, tokenized_sent, senti_label)
                          for caller, tokenized_sent, raw_sent, _, senti_label in session_utts]
            all_lenes.extend([len(tokenized_sent) for caller, tokenized_sent, senti_label in lower_utts])

            # 화자 A와 B의 메타데이터 백터화
            vec_a_age_meta = [0, 0, 0]
            vec_a_age_meta[a_info['age']] = 1
            vec_a_sex_meta = [0, 0]
            vec_a_sex_meta[a_info['sex']] = 1
            vec_a_meta = vec_a_age_meta + vec_a_sex_meta
            vec_b_age_meta = [0, 0, 0]
            vec_b_age_meta[b_info['age']] = 1
            vec_b_sex_meta = [0, 0]
            vec_b_sex_meta[b_info['sex']] = 1
            vec_b_meta = vec_b_age_meta + vec_b_sex_meta

            topic = session_data['topic'] + '_' + a_info['relation_group']
            meta = (vec_a_meta, vec_b_meta, topic)
            dialog = [(bod_utts, 0, None)] + \
                     [(tokenized_sent, int(caller == 'B'), senti_label)
                      for caller, tokenized_sent, senti_label in lower_utts]
            new_utts.extend([bod_utts] + [tokenized_sent for caller, tokenized_sent, senti_label in lower_utts])
            new_dialog.append(dialog)
            new_meta.append(meta)
        return new_dialog, new_meta, new_utts

    def _load_vocab(self, vocab_path):
        with open(vocab_path, 'rb') as reader:
            vocab_file_dict = pickle.load(reader)
        self.vocab = vocab_file_dict['vocab']
        self.rev_vocab = vocab_file_dict['rev_vocab']
        self.unk_id = vocab_file_dict['rev_vocab']['<unk>']
        self.topic_vocab = vocab_file_dict['topic_vocab']
        self.rev_topic_vocab = vocab_file_dict['rev_topic_vocab']
        self.dialog_act_vocab = vocab_file_dict['dialog_act_vocab']
        self.rev_dialog_act_vocab = vocab_file_dict['rev_dialog_act_vocab']

    def _build_vocab(self, max_vocab_count):
        # 대화록 단어
        all_words = []
        for tokens in self.train_corpus[self.utt_id]:
            all_words.extend(tokens)
        # 출현빈도수로 정렬된 단어 사전 정의
        # Counter: 단어 출현 횟수를 딕셔너리로 반환
        # most_common(): 빈도수가 높은 순서대로 튜플로 반환
        vocab_count = Counter(all_words).most_common()[0:max_vocab_count]
        self.vocab = self.reserved_token_for_gen + [token for token, count in vocab_count]
        self.rev_vocab = {token: idx for idx, token in enumerate(self.vocab)}
        self.unk_id = self.rev_vocab['<unk>']

        # 대화록 토픽
        all_topics = [topic for _, _, topic in self.train_corpus[self.meta_id]]
        # 토픽이 등장 빈도수 순서로 저장되어 있는 리스트
        self.topic_vocab = [topic for topic, _ in Counter(all_topics).most_common()]
        # 토픽에 인덱스가 부여된 딕셔너리
        self.rev_topic_vocab = {topic: idx for idx, topic in enumerate(self.topic_vocab)}

        # 대화록 감정분석
        all_sentiments = []
        for dialog in self.train_corpus[self.dialog_id]:
            all_sentiments.extend(
                [senti_label for _, _, senti_label in dialog if senti_label is not None]
            )
        # 감정 레이블이 등장 빈도수 순서로 저장되어 있는 리스트
        self.dialog_act_vocab = [senti_label for senti_label, _ in Counter(all_sentiments).most_common()]
        # 감정 레이블에 인덱스가 부여된 딕셔너리
        self.rev_dialog_act_vocab = {senti_label: idx for idx, senti_label in enumerate(self.dialog_act_vocab)}

    def _save_vocab(self, vocab_path):
        with open(vocab_path, 'wb') as writer:
            pickle.dump(
                {
                    'vocab': self.vocab,
                    'rev_vocab': self.rev_vocab,
                    'topic_vocab': self.topic_vocab,
                    'rev_topic_vocab': self.rev_topic_vocab,
                    'dialog_act_vocab': self.dialog_act_vocab,
                    'rev_dialog_act_vocab': self.rev_dialog_act_vocab
                }, writer
            )

    def _load_word2vec(self, word2vec_path, word2vec_dim):
        if not os.path.exists(word2vec_path):
            return

        self.word2vec = []
        raw_word2vec = FastText.load_fasttext_format(word2vec_path)
        reserved_tokens = self.reserved_token_for_dialog + self.reserved_token_for_gen
        oov_cnt = 0
        for vocab in self.vocab:
            if vocab == '<pad>':
                vec = np.zeros(word2vec_dim)
            elif vocab in reserved_tokens:
                vec = np.random.randn(word2vec_dim) * 0.1
            else:
                if vocab in raw_word2vec:
                    vec = raw_word2vec[vocab]
                else:
                    oov_cnt += 1
                    vec = np.random.randn(word2vec_dim) * 0.1
            self.word2vec.append(vec)

    def get_dialog_corpus(self):
        # 토크나이징 된 단어들을 토큰 아이디로 변경
        def _word_to_idx(data):
            result = []
            for dialog in data:
                temp = []
                for tokenized_sent, caller, senti_label in dialog:
                    senti_label_idx = self.rev_dialog_act_vocab[senti_label] if senti_label is not None else None
                    temp.append(([self.rev_vocab.get(token, self.unk_id) for token in tokenized_sent], caller, senti_label_idx))
                result.append(temp)
            return result
        return {
            'train': _word_to_idx(self.train_corpus[self.dialog_id]),
            'test': _word_to_idx(self.test_corpus[self.dialog_id]),
            'valid': _word_to_idx(self.valid_corpus[self.dialog_id])
        }

    def get_meta_corpus(self):
        def _word_to_idx(data):
            result = []
            for vec_a_meta, vec_b_meta, topic in data:
                result.append((vec_a_meta, vec_b_meta, self.rev_topic_vocab[topic]))
            return result
        return {
            'train': _word_to_idx(self.train_corpus[self.meta_id]),
            'test': _word_to_idx(self.test_corpus[self.meta_id]),
            'valid': _word_to_idx(self.valid_corpus[self.meta_id])
        }
