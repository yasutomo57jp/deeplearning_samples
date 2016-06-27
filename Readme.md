# Deep Learning勉強用のサンプルコード

できるだけシンプルなサンプルコードを書く試み

## test_caffemodel.py
### 何をするコード？
ImageNetで学習した1000クラス分類用caffeのモデルを読み込んで画像認識をするサンプル

### 必要な準備

caffemodelと，ImageNetの平均画像及び1000クラス分類のクラスラベルをダウンロードしておく

```bash
wget http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel
wget https://github.com/BVLC/caffe/raw/master/python/caffe/imagenet/ilsvrc_2012_mean.npy
wget http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz
tar zxvf caffe_ilsvrc12.tar.gz
```

テスト用の画像を適当に準備し，255x255のカラー画像にしておく．
ファイル名は testimage.jpg にする．

### 実行方法
```bash
python test_caffemodel.py
```
## feature_extraction.py
### 何をするコード？
ImageNetで学習した1000クラス分類用のcaffeモデルを元に，7層目から4096次元の特徴量を抽出するサンプル

### 必要な準備

caffemodelと，ImageNetの平均画像及び1000クラス分類のクラスラベルをダウンロードしておく

```bash
wget http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel
wget https://github.com/BVLC/caffe/raw/master/python/caffe/imagenet/ilsvrc_2012_mean.npy
wget http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz
tar zxvf caffe_ilsvrc12.tar.gz
```

特徴抽出したい画像ファイルを適当なディレクトリに置く（複数可）．

### 実行方法
```bash
python feature_extraction.py imagesdir feature.pickle
```

[(画像ファイル名1,特徴量1), (画像ファイル名2,特徴量2), ...] のデータがpickleに書き出されます．
