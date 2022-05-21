from time import time
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, Embedding, LSTM, GRU, Conv2D, GlobalMaxPool1D, Dense, Dropout

from extraction import embedding_w2v
from extraction import split_and_zero_padding
from extraction import ManhatDist


data_train_file = './data/predictest1.csv'

train_df = pd.read_csv(data_train_file)
for a in ['answer', 'key_answer']:
    train_df[a + '_n'] = train_df[a]

embedding_dim = 300
max_seq_lenght = 20
use_w2v = True

train_df, embeddings = embedding_w2v(train_df, embedding_dim=embedding_dim, empty_w2v=not use_w2v)

validation_size = int(len(train_df) * 0.1)
training_size = len(train_df) - validation_size

X = train_df[['answer_n', 'key_answer_n']]
Y = train_df['is_duplicate']
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)

X_train = split_and_zero_padding(X_train, max_seq_lenght)
X_validation = split_and_zero_padding(X_validation, max_seq_lenght)

Y_train = Y_train.values
Y_validation = Y_validation.values

assert X_train['left'].shape == X_train['right'].shape
assert len(X_train['left']) == len(Y_train)

gpus = 2
batch_size = 1024 * gpus
n_epoch = 50
n_hidden = 50

sm = Sequential()
sm.add(Embedding(len(embeddings), embedding_dim, 
                weights=[embeddings], input_shape=(max_seq_lenght,), trainable=False))

sm.add(LSTM(n_hidden))
shared_model = sm

left_input = Input(shape=(max_seq_lenght,), dtype='int32')
right_input = Input(shape=(max_seq_lenght,), dtype='int32')

malstm_distance = ManhatDist()([shared_model(left_input), shared_model(right_input)])
model = Model(inputs=[left_input, right_input], outputs=[malstm_distance])

if  gpus >=2:
    model = tf.keras.utils.multi_gpu_model(model, gpus=gpus)
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
model.summary()
shared_model.summary()

training_start_time = time()
malstm_trained = model.fit([X_train['left'], X_train['right']], Y_train,
                            batch_size=batch_size, epochs=n_epoch,
                            validation_data=([X_validation['left'], X_validation['right']],Y_validation))
training_end_time = time()

model.save('./data/SiameseLSTM.h5')

plt.subplot(211)
plt.plot(malstm_trained.history['acc'])
plt.plot(malstm_trained.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')


plt.subplot(212)
plt.plot(malstm_trained.history['loss'])
plt.plot(malstm_trained.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.tight_layout(h_pad=1.0)
plt.savefig('./data/history-graph.png')

print(str(malstm_trained.history['val_acc'][-1])[:6] +
      "(max: " + str(max(malstm_trained.history['val_acc']))[:6] + ")")
print("Done.")