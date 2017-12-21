# -*- coding: utf-8 -*-

import time
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.externals import joblib
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Dense, Activation, Dropout

# 学習データとテストデータに分ける
data_train, data_test, label_train, label_test = joblib.load("mnist")
data_train = np.asarray(data_train, np.float32)
data_test = np.asarray(data_test, np.float32)
label_train = np.asarray(label_train, np.int32)
label_test = np.asarray(label_test, np.int32)


# 学習データを画像に変換
def conv_feat_2_image(feats):
    data = np.ndarray((len(feats), 1, 28, 28), dtype=np.float32)
    for i, f in enumerate(feats):
        data[i] = f.reshape(28, 28)
    return data

data_train = conv_feat_2_image(data_train)
data_test = conv_feat_2_image(data_test)

data_train /= np.max(data_train)
data_test /= np.max(data_test)

label_test_binary = to_categorical(label_test)
label_train_binary = to_categorical(label_train)

model = Sequential()

model.add(Conv2D(32, (3, 3), border_mode='same', input_shape=(1, 28, 28)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(3, 3), border_mode='same'))

model.add(Conv2D(64, (3, 3), border_mode='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(3, 3), border_mode='same'))

model.add(Flatten())

model.add(Dense(200))
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Dense(100))
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Dense(10))
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

start = time.time()  # 処理時間の計測開始
training = model.fit(data_train, label_train_binary,
                     epochs=100, batch_size=100, verbose=1)
training_time = time.time() - start
with open("classifiers/keras_cnn", "wb") as fout:
    import pickle
    pickle.dump((model.to_json(), training_time, training.history["loss"][-1]),
                fout)
model.save_weights('classifiers/keras_cnn.hdf5')


# 評価
start = time.time()
results = list(model.predict_classes(data_test, verbose=1))
predict_time = time.time() - start

# %%
# 認識率を計算
score = accuracy_score(label_test, results)
print()
print(training_time, predict_time)
print(score)
cmatrix = confusion_matrix(label_test, results)
print(cmatrix)
with open("results/keras_cnn", "wb") as fout:
    import pickle
    pickle.dump((training_time, predict_time, score, cmatrix), fout)
