# -*- coding: utf-8 -*-
import time
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.externals import joblib
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout


# 学習データとテストデータに分ける
data_train, data_test, label_train, label_test = joblib.load("mnist")
data_train = np.asarray(data_train, np.float32)
data_test = np.asarray(data_test, np.float32)
label_train = np.asarray(label_train, np.int32)
label_test = np.asarray(label_test, np.int32)

label_test_binary = to_categorical(label_test)
label_train_binary = to_categorical(label_train)

model = Sequential()

model.add(Dense(200, input_dim=784))
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
with open("classifiers/keras_nn", "wb") as fout:
    import pickle
    pickle.dump((model.to_json(), training_time, training.history["loss"][-1]),
                fout)

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
with open("results/keras_nn", "wb") as fout:
    import pickle
    pickle.dump((training_time, predict_time, score, cmatrix), fout)
