import pandas as pd


from tensorflow.python import keras

from extraction import embedding_w2v
from extraction import split_and_zero_padding
from extraction import ManhatDist

# path data untuk predict
data_file = './data/df-test.csv'

# memuat data dari file
test_df = pd.read_csv(data_file)
for a in ['answer1', 'answer2']:
    test_df[a + '_n'] = test_df[a]

# membuat word2vec embedding
embedding_dim = 300
max_seq_length = 20
test_df, embeddings = embedding_w2v(test_df, embedding_dim=embedding_dim, empty_w2v=False)

# memisahkan data jawaban dan memberikan zero padding
X_test = split_and_zero_padding(test_df, max_seq_length)

# memastikan ukuran dan bentuk data jawaban sama
assert X_test['left'].shape == X_test['right'].shape

# memuat model hasil training 
model = keras.models.load_model('./data/SiameseLSTM.h5', custom_objects={'ManhatDist': ManhatDist})
model.summary()

# mulai prediksi data
prediction = model.predict([X_test['left'], X_test['right']])


# print hasil prediksi
for n in prediction:
    print(str(n))
