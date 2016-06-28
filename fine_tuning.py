#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 使い方：
#   画像をあるフォルダに保存しておく(images)

from __future__ import print_function
import sys
import os
import os.path
import pickle
import cv2
import numpy as np
from chainer import Variable
from chainer.functions import caffe
from chainer import cuda
from chainer import Chain
import chainer.links as L
import chainer.functions as F
from chainer import optimizers

if len(sys.argv) < 4:
    print("usage: %s imagedir newlabels outputfile" % sys.argv[0])
    quit()
inputpath = sys.argv[1]
newlabelfile = sys.argv[2]
outputfile = sys.argv[3]

# Caffeのモデル読み込み（時間がかかる）
print("loading model ... ", end="", file=sys.stderr)
model = caffe.CaffeFunction("bvlc_reference_caffenet.caffemodel")
model.to_gpu()
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


def load_newlabels(newlabelfile, filenames):
    # カンマ区切り
    # ファイル名,ラベル(int) の形式

    with open(newlabelfile) as fin:
        lines = fin.readlines()
    labels = dict(map(lambda y: (y[0], int(y[1])), map(lambda x: x.strip().split(","), lines)))

    return [labels[fn] for fn in filenames]

print("loading images ... ", end="", file=sys.stderr)
trainingimages, filenames = load_images(inputpath, mean)
print("done", file=sys.stderr)

print("loading new labels ... ", end="", file=sys.stderr)
labels = load_newlabels(newlabelfile, filenames)
print("done", file=sys.stderr)

class BinClassNet(Chain):
    def __init__(self, model):
        super(BinClassNet, self).__init__(
                base = model,
                hlayer = L.Linear(4096, 2000),
                binlayer = L.Linear(2000, 2))
        self.train = True

    def __call__(self, x):
        h1 = self.base(inputs={"data": x}, outputs=["fc7"], train=self.train)
        h2 = F.relu(self.hlayer(h1[0]))
        h3 = self.binlayer(h2)
        return h3


net = BinClassNet(model)
clf = L.Classifier(net)
clf.compute_accuracy = True
clf.to_gpu()

optimizer = optimizers.Adam()
optimizer.setup(clf)

epoch = 100
batchsize = 10
results = []
N = len(trainingimages)
for e in range(epoch):
    sum_loss = 0.0
    sum_accuracy = 0.0
    for i in range(0, N, batchsize):
        # chainer用の変数にする
        x = Variable(cuda.cupy.asarray(trainingimages[i:i+batchsize]))
        t = Variable(cuda.cupy.asarray(labels[i:i+batchsize], dtype=np.int32))

        optimizer.zero_grads()

        loss = clf(x, t)
        accuracy = clf.accuracy

        loss.backward()

        optimizer.update()

        sum_loss += float(loss.data) * batchsize
        sum_accuracy += float(accuracy.data) * batchsize

    print("loss: %f, accuracy: %f" % (sum_loss / N, sum_accuracy / N))
