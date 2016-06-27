#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 使い方：
#   画像をあるフォルダに保存しておく(images)

from __future__ import print_function
import sys
import os
import os.path
import cv2
import numpy as np
from chainer import Variable
from chainer.functions import caffe
from chainer import cuda

if len(sys.argv) > 3:
    print("usage: %s imagedir outputfile" % sys.argv[0])
    quit()
inputpath = sys.argv[1]
outputfile = sys.argv[2]

# Caffeのモデル読み込み（時間がかかる）
print("loading model ... ", end="", file=sys.stderr)
model = caffe.CaffeFunction("bvlc_reference_caffenet.caffemodel")
model.to_gpu()
print("done", file=sys.stderr)

# ラベルの読み込み
print("loading labels ... ", end="", file=sys.stderr)
with open("synset_words.txt") as fin:
    labels = fin.readlines()
print("done", file=sys.stderr)

# 平均画像の読み込み
mean = np.load("ilsvrc_2012_mean.npy")  # 3x255x255 の画像


def load_images(inputpath, mean):
    imglist = []
    filenames = []

    for root, dirs, files in os.walk(inputpath):
        for fn in sorted(files):
            filenames.append(fn)
            bn, ext = os.path.splitext(fn)
            if ext not in [".bmp", ".jpg", ".png"]:
                continue

            filename = os.path.join(root, fn)
            # 評価用画像の読み込み（255x255サイズのカラー画像）
            # チャンネル，高さ，幅に入れ替え
            testimage = np.asarray(cv2.imread(filename), dtype=np.float64)
            testimage = cv2.resize(testimage, (256,256))
            testimage = testimage.transpose(2, 0, 1)

            # 平均を引いた画像
            testimage = testimage - mean

            # 画像サイズを bvlc_reference_caffenet 用の 3x227x227 に揃えて配列にする
            start = (255 - 227) // 2
            stop = start + 227

            imglist.append(testimage[:, start:stop, start:stop])
    imgsdata = np.asarray(imglist, dtype=np.float32)

    return imgsdata, filenames

print("loading images ... ", end="", file=sys.stderr)
testimages, filenames = load_images(inputpath, mean)
print("done", file=sys.stderr)

batchsize = 10
results = []
for i in range(0, len(testimages), batchsize):
    # chainer用の変数にする
    x = Variable(cuda.cupy.asarray(testimages[i:i+batchsize]))

    # ネットワークを通す
    y = model(inputs={"data": x}, outputs=["fc8"], train=False)

    # 結果を受け取る
    outputs = cuda.to_cpu(y[0].data)  # 1000クラスそれぞれのスコア
    class_ids = np.argmax(outputs, axis=1)  # 最大スコアのクラス番号を返す

    for j, class_id in enumerate(class_ids):
        print(filenames[i+j], labels[class_id])
        results.append((filenames[i+j], labels[class_id]))

with open(outputfile, "wb") as fout:
    pickle.dump(results, fout)
