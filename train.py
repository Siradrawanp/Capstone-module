from time import time
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.python.keras import callbacks
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, Embedding, LSTM

from extraction import embedding_w2v
from extraction import split_and_zero_padding
from extraction import ManhatDist

# file path data
data_train_file = './data/df-train.csv'

# memuat file data
train_df = pd.read_csv(data_train_file)
for a in ['answer1', 'answer2']:
    train_df[a + '_n'] = train_df[a]

# membuat embedding word2vec
embedding_dim = 300
max_seq_lenght = 20
use_w2v = True

train_df, embeddings = embedding_w2v(train_df, embedding_dim=embedding_dim, empty_w2v=not use_w2v)

# memisahkan data untuk validasi dan training
validation_size = int(len(train_df) * 0.1)
training_size = len(train_df) - validation_size

X = train_df[['answer1', 'answer2']]
Y = train_df['is_duplicate']
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)

print(X_train)


X_train = split_and_zero_padding(X_train, max_seq_lenght)
X_validation = split_and_zero_padding(X_validation, max_seq_lenght)

# Konversi label dalam representasi numpy
Y_train = Y_train.values
Y_validation = Y_validation.values

# memastikan bentuk dan ukuran kolom jawaban sama
assert X_train['left'].shape == X_train['right'].shape
assert len(X_train['left']) == len(Y_train)

# variabel untuk model
gpus = 2
batch_size = 1024 * gpus
n_epoch = 50
n_hidden = 40

# mendefinisikan callback untuk epoch pada training
earlystop = callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=5, restore_best_weights=True)

# mendefinisikan shared model
sm = Sequential()
sm.add(Embedding(len(embeddings), embedding_dim, 
                weights=[embeddings], input_shape=(max_seq_lenght,), trainable=False))

sm.add(LSTM(n_hidden))
shared_model = sm

# inisialisasi layer yang terlihat (visible layer)
left_input = Input(shape=(max_seq_lenght,), dtype='int32')
right_input = Input(shape=(max_seq_lenght,), dtype='int32')

# memasukan Manhattan Distance model
malstm_distance = ManhatDist()([shared_model(left_input), shared_model(right_input)])
model = Model(inputs=[left_input, right_input], outputs=[malstm_distance])

#if  gpus >=2:
#    model = tf.keras.utils.multi_gpu_model(model, gpus=gpus)

model.compile(loss='mean_squared_error', optimizer='adam' , metrics=['accuracy'])
model.summary()
shared_model.summary()

# memulai training data
training_start_time = time()
malstm_trained = model.fit([X_train['left'], X_train['right']], Y_train,
                            batch_size=batch_size, epochs=n_epoch,
                            validation_data=([X_validation['left'], X_validation['right']],Y_validation),
                            callbacks=[earlystop])
training_end_time = time()

# menyimpan model hasil training
model.save('./data/SiameseLSTM.h5')

# plotting akurasi hasil training
plt.subplot(211)
plt.plot(malstm_trained.history['accuracy'])
plt.plot(malstm_trained.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# plotting loss data training
plt.subplot(212)
plt.plot(malstm_trained.history['loss'])
plt.plot(malstm_trained.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

# menyimpan gambar plotting
plt.tight_layout(h_pad=1.0)
plt.savefig('./data/history-graph.png')

# ouput akurasi validasi
print(str(malstm_trained.history['val_accuracy'][-1])[:6] +
      "(max: " + str(max(malstm_trained.history['val_accuracy']))[:6] + ")")
print("Done.")
'''
'''