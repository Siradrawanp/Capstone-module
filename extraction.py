
from gensim.models import KeyedVectors



from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer
from keras.utils.data_utils import pad_sequences

import numpy as np
import itertools
import string

from nltk.corpus import stopwords

import yake


def keyword_extraction(text):

    # ekstraksi kata kunci dengan yake
    kw_extractor = yake.KeywordExtractor(lan="id", n=5, stopwords="id", top=20)
    keywords = kw_extractor.extract_keywords(text)

    final_kw = []

    for i in keywords:
        kw = next(iter(i))
        kw = str(kw)
        final_kw.append(kw)
    

    # membersihkan tanda baca pada kata kunci yang telah diekstrak
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in final_kw]    

    return stripped


def embedding_w2v(df, embedding_dim=300, empty_w2v=False):
    vocabs = {}
    vocabs_count = 0

    vocabs_not_w2v = {}
    vocabs_not_w2v_count = 0

    # menyetel stopword 
    stop_words = set(stopwords.words('indonesian'))

    # memuat model word2vector google
    if empty_w2v:
        word2vec = EmptyWord2Vec
    else:
        word2vec = KeyedVectors.load_word2vec_format("./data/GoogleNews-vectors-negative300.bin.gz", binary=True)

    for index, row in df.iterrows():
        # mencetak jumlah kalimat yang telah disematkan (embedding)
        if  index != 0 and index % 50 == 0:
            print("{:,} sentences embedded.", format(index), flush=True)

        # iterasi kata pada tiap jawaban
        for answer in ['answer1', 'answer2']:
            
            ka2 = []
            for key_word in keyword_extraction(row[answer]):
                # memeriksa kata kunci termasuk dalam daftar stopword
                if key_word in stop_words:
                    continue

                # counter kata yang tidak terdapat pada model word2vec google
                if key_word not in word2vec:
                    vocabs_not_w2v_count += 1
                    vocabs_not_w2v[key_word] = 1

                # menambahkan kata yang tidak terdapat pada model word2vec google
                if key_word not in vocabs:
                    vocabs_count += 1
                    vocabs[key_word] = vocabs_count
                    ka2.append(vocabs_count)
                else:
                    ka2.append(vocabs[key_word])

                # menetapkan index dari kata yang tidak terdapat pada model word2vec google
                df._set_value(index, answer, ka2)
    
    # matrik embedding
    embeddings = 1 * np.random.randn(len(vocabs) + 1, embedding_dim)
    embeddings[0] = 0   # untuk menghiraukan padding 0

    # membuat matrik embedding
    for key_word, index in vocabs.items():
        if key_word in word2vec:
            embeddings[index] = word2vec.word_vec(key_word)
    del word2vec

    return df, embeddings

def split_and_zero_padding(df, max_seq_lenght):
    # memisahkan berdasarkan dict/jawaban
    X = {'left': df['answer1'], 'right': df['answer2']}

    # zero padding
    for dataset, side in itertools.product([X], ['left', 'right']):
        dataset[side] = pad_sequences(dataset[side], padding = 'pre', truncating = 'post', maxlen = max_seq_lenght)

    return dataset

class ManhatDist(Layer):

    # Keras custom layer untuk menghitung Manhattan Distance

    # inisialisasi layer
    def __init__(self, **kwargs):
        self.result = None
        super(ManhatDist, self).__init__(**kwargs)

    # membuat layer dengan masukan input_shape
    def build(self, input_shape):
        super(ManhatDist, self).build(input_shape)

    # logic atau rumus persamaan dari manhattan distance
    def call(self, x, **kwargs):
        self.result = K.exp(-K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True))
        return self.result

    # mengembalikan output shape
    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)

class EmptyWord2Vec:
    vocab = {}
    word_vec = {}