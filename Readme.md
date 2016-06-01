# Deep Learning勉強用のサンプルコード

できるだけシンプルなサンプルコードを書く試み

## test_caffemodel.py
### 何をするコード？
ImageNetで学習した1000クラス分類用caffeのモデルを読み込んで画像認識をするサンプル

### 必要な準備

caffemodelと，ImageNetの平均画像及び1000クラス分類のクラスラベルをダウンロードしておく

* wget http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel
* wget https://github.com/BVLC/caffe/raw/master/python/caffe/imagenet/ilsvrc_2012_mean.npy
* wget http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz
* tar zxvf caffe_ilsvrc12.tar.gz

テスト用の画像を適当に準備し，255x255のカラー画像にしておく．
ファイル名は testimage.jpg にする．

### 実行方法

python test_caffemodel.py


