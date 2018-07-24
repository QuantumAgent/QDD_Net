import numpy as np
import pandas as pd
from collections import Counter
import pickle

lmap = lambda func, it: list(map(lambda x: func(x), it))

TRAIN_PATH = 'dataset/train.csv'
TEST_PATH = 'dataset/test.csv'
QUESTION_PATH = 'dataset/question.csv'
WORD_EMBED = 'dataset/word_embed.txt'
CHAR_EMBED = 'dataset/char_embed.txt'


def get_ids(qids):
    ids = []
    for t_ in qids:
        ids.append(int(t_[1:]))
    return np.asarray(ids)


def get_texts(file_path, question_path):
    qes = pd.read_csv(question_path)
    file = pd.read_csv(file_path)
    q1id, q2id = file['q1'], file['q2']
    id1s, id2s = get_ids(q1id), get_ids(q2id)
    all_words = qes['words']
    texts = []
    for t_ in zip(id1s, id2s):
        texts.append(all_words[t_[0]] + ' ' + all_words[t_[1]])
    return texts


def crop_pad(max_length, word_index):
    if len(word_index) > max_length:
        return word_index[:max_length]
    pad_length = max_length - len(word_index)
    word_index = word_index + [0] * pad_length
    assert len(word_index) == max_length
    return word_index


def main():
    print("Load files...")
    questions = pd.read_csv(QUESTION_PATH)
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    with open(WORD_EMBED, 'r+') as f:
        embedding_string = f.readlines()
    with open(CHAR_EMBED, 'r+') as f:
        char_embedding_string = f.readlines()

    print("Writing word/char file")
    word_id = Counter()
    word_matrix = []
    word_id.setdefault('', 0)
    word_matrix.append([0] * 300)
    for e in embedding_string:
        es = e.strip().split()
        word_id.setdefault(es[0], len(word_id))
        word_matrix.append(lmap(lambda x: float(x), es[1:]))

    char_id = Counter()
    char_matrix = []
    char_id.setdefault('', 0)
    char_matrix.append([0] * 300)
    for e in char_embedding_string:
        es = e.strip().split()
        char_id.setdefault(es[0], len(char_id))
        char_matrix.append(lmap(lambda x: float(x), es[1:]))

    word_matrix = np.array(word_matrix)
    char_matrix = np.array(char_matrix)

    np.save('word_matrix', word_matrix)
    np.save('char_matrix', char_matrix)
    with open('word_index.pkl', 'wb+') as f:
        pickle.dump(word_id, f)
    with open('char_index.pkl', 'wb+') as f:
        pickle.dump(char_id, f)

    print("Writing training/developing/testing sets")

    questions.index = questions.qid
    np.max(questions['words'].apply(lambda x: len(x.split())))
    np.max(questions['chars'].apply(lambda x: len(x.split())))

    train_set = []
    for k, v in train.iterrows():
        q1_id = v['q1']
        q2_id = v['q2']
        label = v['label']
        q1_words = lmap(lambda x: word_id[x], questions.loc[q1_id]['words'].split())
        q1_words = crop_pad(39, q1_words)
        q1_chars = lmap(lambda x: char_id[x], questions.loc[q1_id]['chars'].split())
        q1_chars = crop_pad(58, q1_chars)
        q2_words = lmap(lambda x: word_id[x], questions.loc[q2_id]['words'].split())
        q2_words = crop_pad(39, q2_words)
        q2_chars = lmap(lambda x: char_id[x], questions.loc[q2_id]['chars'].split())
        q2_chars = crop_pad(58, q2_chars)
        train_set.append((q1_words, q1_chars, q2_words, q2_chars, label))

    q1w = np.array(lmap(lambda x: x[0], train_set))
    q1c = np.array(lmap(lambda x: x[1], train_set))
    q2w = np.array(lmap(lambda x: x[2], train_set))
    q2c = np.array(lmap(lambda x: x[3], train_set))
    y = np.expand_dims(np.array(lmap(lambda x: x[4], train_set)), axis=1)

    train_corpus = np.concatenate((q1w, q1c, q2w, q2c, y), axis=-1)
    np.save('train_corpus', train_corpus)

    test_set = []
    for k, v in test.iterrows():
        q1_id = v['q1']
        q2_id = v['q2']
        q1_words = lmap(lambda x: word_id[x], questions.loc[q1_id]['words'].split())
        q1_words = crop_pad(39, q1_words)
        q1_chars = lmap(lambda x: char_id[x], questions.loc[q1_id]['chars'].split())
        q1_chars = crop_pad(58, q1_chars)
        q2_words = lmap(lambda x: word_id[x], questions.loc[q2_id]['words'].split())
        q2_words = crop_pad(39, q2_words)
        q2_chars = lmap(lambda x: char_id[x], questions.loc[q2_id]['chars'].split())
        q2_chars = crop_pad(58, q2_chars)
        test_set.append((q1_words, q1_chars, q2_words, q2_chars))

    q1wt = np.array(lmap(lambda x: x[0], test_set))
    q1ct = np.array(lmap(lambda x: x[1], test_set))
    q2wt = np.array(lmap(lambda x: x[2], test_set))
    q2ct = np.array(lmap(lambda x: x[3], test_set))

    test_corpus = np.concatenate((q1wt, q1ct, q2wt, q2ct), axis=-1)
    np.save('test_corpus', test_corpus)


if __name__ == '__main__':
    main()
