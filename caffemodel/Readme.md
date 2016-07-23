# caffemodel

## 必要な準備

caffemodelと，ImageNetの平均画像及び1000クラス分類のクラスラベルをダウンロードしておく

```bash
wget http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel
wget https://github.com/BVLC/caffe/raw/master/python/caffe/imagenet/ilsvrc_2012_mean.npy
wget http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz
tar zxvf caffe_ilsvrc12.tar.gz
```

## classify_one.py

ImageNetで学習した1000クラス分類用caffeのモデルを読み込んで画像認識をするサンプル

テスト用の画像を適当に準備し，255x255のカラー画像にしておく．
ファイル名は testimage.jpg にする．

### 実行方法

```bash
python classify_one.py
```

## classify_images.py

ImageNetで学習した1000クラス分類用caffeのモデルを読み込んで複数の画像に対して認識をするサンプル

テスト用の画像を適当に準備し，適当なディレクトリに保存しておく．

### 実行方法

```bash
python classify_images.py imagedir outputfile
```

## feature_extraction.py

ImageNetで学習した1000クラス分類用のcaffeモデルを元に，7層目から4096次元の特徴量を抽出するサンプル

特徴抽出したい画像ファイルを適当なディレクトリに保存しておく．

### 実行方法
```bash
python feature_extraction.py imagesdir feature.pickle
```

[(画像ファイル名1,特徴量1), (画像ファイル名2,特徴量2), ...] のデータがpickleに書き出されます．

## fine_tuning.py

ImageNetで学習した1000クラス分類用のcaffeモデルを元に，7層目から4096次元の特徴量を抽出し，
fine tuningをして2クラス分類器を学習するサンプル

特徴抽出したい画像ファイルを適当なディレクトリに保存しておく．

### 実行方法
```bash
python feature_extraction.py imagesdir newlabels.txt feature.pickle
```

newlabels.txt には，画像のファイル名とラベル(int)をカンマ区切りで書いておく．

```
image1.png,0
image2.png,0
image3.png,1
image4.png,1
```
