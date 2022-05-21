import gensim
from gensim.models import KeyedVectors

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.utils.data_utils import pad_sequences

import numpy as np
import itertools

from nltk.corpus import stopwords

import yake


def keyword_extraction(text):
    kw_extractor = yake.KeywordExtractor(lan="id", n=3, stopwords="id", top=50)
    keywords = kw_extractor.extract_keywords(text)

    final_kw = []

    for i in keywords:
        kw = next(iter(i))
        kw = str(kw)
        final_kw.append(kw)
    
    return final_kw



def embedding_w2v(df, embedding_dim=300, empty_w2v=False):
    vocabs = {}
    vocabs_count = 0

    vocabs_not_w2v = {}
    vocabs_not_w2v_count = 0

    stop_words = set(stopwords.words('indonesian'))

    if empty_w2v:
        word2vec = EmptyWord2Vec
    else:
        word2vec = KeyedVectors.load_word2vec_format("./data/GoogleNews-vectors-negative300.bin.gz", binary=True)

    for index, row in df.iterrows():
        if  index != 0 and index % 1000 == 0:
            print("{:,} sentences embedded.", format(index), flush=True)

        for answer in ['answer', 'key_answer']:
            
            ka2 = []
            for key_word in keyword_extraction(row[answer]):
                if key_word in stop_words:
                    continue

                if key_word not in word2vec.vocab:
                    vocabs_not_w2v_count += 1
                    vocabs_not_w2v[key_word] = 1

                if key_word not in vocabs:
                    vocabs_count += 1
                    vocabs[key_word] = vocabs_count
                    ka2.append(vocabs_count)
                else:
                    ka2.append(vocabs[key_word])

                df.at[index, answer + '_n'] = ka2
    
    embeddings = 1 * np.random.rand(len(vocabs)+1, embedding_dim)
    embeddings[0] = 0

    for key_word, index in vocabs.items():
        if key_word in word2vec.vocab:
            embeddings[index] = word2vec.word_vec(key_word)
    del word2vec

    return df, embeddings

def split_and_zero_padding(df, max_seq_lenght):
    X = {'left': df['answer_n'], 'right': df['key_answer_n']}

    for dataset, side in itertools.product([X], ['left', 'right']):
        dataset[side] = pad_sequences(dataset[side], padding = 'pre', truncating = 'post', maxlen = max_seq_lenght)

    return dataset

class ManhatDist(Layer):
    def __init__(self, **kwargs):
        self.result = None
        super(ManhatDist, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ManhatDist, self).build(input_shape)

    def call(self, x, **kwargs):
        self.result = K.exp(-K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True))
        return self.result

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)

class EmptyWord2Vec:
    vocab = {}
    word_vec = {}