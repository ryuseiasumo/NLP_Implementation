# NLP_Implementation  
Pytorchで自然言語処理のモデルを実装してみました。
* CBOW(cbow)  
* BERTを用いたテキスト分類モデル(sentence_classification)  

# 開発環境
* python 3.7.10
* pytorch 1.3.1

# Requirement
* python
* pytorch



# 使用方法例(sentence_classification)
```bash
$ git clone https://github.com/ryuseiasumo/NLP_Implementation.git
$ cd NLP_Implementation
$ cd sentence_classification
$ python main.py --use_data Livedoor --batch_size 10 --max_epoch 100
```

# 使用方法例(cbow)
```bash
$ git clone https://github.com/ryuseiasumo/NLP_Implementation.git
$ cd NLP_Implementation
$ cd cbow
$ python main.py --use_data Sample --window_size 2 --vocab_size 50 --emb_dim 15 --batch_size 10 --max_epoch 100
```

# Note
* sentence_classificationの方はCPUだと学習にかなり時間がかかるので、動作を確認したい方は以下のように、main.pyの69行目をコメントアウトし、70行目のコメントを解除して試して下さい。
```bash
#name = "all.tsv" #全データ
name = "train.tsv" #データ
```
