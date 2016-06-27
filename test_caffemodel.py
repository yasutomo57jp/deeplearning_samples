#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from chainer import Variable
from chainer.functions import caffe
from chainer import cuda

# Caffeのモデル読み込み（時間がかかる）
model = caffe.CaffeFunction("bvlc_reference_caffenet.caffemodel")
model.to_gpu()

# ラベルの読み込み
with open("synset_words.txt") as fin:
    labels = fin.readlines()

# 平均画像の読み込み
mean = np.load("ilsvrc_2012_mean.npy")  # 3x255x255 の画像

# 評価用画像の読み込み（255x255サイズのカラー画像）
testimage = cv2.imread("testimage.jpg").transpose(2, 0, 1)  # チャンネル，高さ，幅に入れ替え

# 平均を引いた画像
img = testimage - mean

# 画像サイズを bvlc_reference_caffenet 用の 3x227x227 に揃えて配列にする
start = (255 - 227) // 2
stop = start + 227
imgsdata = np.asarray([img[:, start:stop, start:stop]], dtype=np.float32)

# chainer用の変数にする
x = Variable(cuda.cupy.asarray(imgsdata))

# ネットワークを通す
y = model(inputs={"data": x}, outputs=["fc8"], train=False)

# 結果を受け取る
outputs = cuda.to_cpu(y[0].data)  # 1000クラスそれぞれのスコア
class_id = np.argmax(outputs)  # 最大スコアのクラス番号を返す

print(labels[class_id])  # クラス名を出力
