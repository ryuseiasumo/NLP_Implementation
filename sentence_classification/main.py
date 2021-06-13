import torch
import torch.nn as nn
import argparse

import torch.nn.functional as F
from torch import optim

from tqdm import trange, tqdm

from src import sentence_classification_model

from src.data.data_module import LivedoorDataModule

from src.train import Trainer
from src.train import Early_Stopping
from src.test import Tester
from src.out_prediction import Out_prediction

from pathlib import Path
SWD = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--use_data', dest='use_data', help='which data use(Sample, Livedoor)', default="Sample", type=str)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size', default=10, type=int)
    parser.add_argument('--max_epoch', dest='max_epoch', help='Num of max epoch', default=100, type=int)

    args = parser.parse_args()
    return args

args = parse_args()
num_epochs = args.max_epoch
batch_size = args.batch_size

import os
import random
import numpy as np
def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def set_model(label_dim):
    net = sentence_classification_model.SentenceClassification(label_dim)
    # GPUを使うか判定
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    net.to(device)

    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()

    return net, optimizer, loss_function



if __name__ == "__main__":
    seed_everything(0) #シード値の初期化
    if args.use_data == "Livedoor":
        #livedoorニュースによる学習
        corpus_dir = Path("corpus/livedoor")
        name = "all.tsv" #全データ
        # name = "train.tsv" #データ
        corpus_path = SWD /corpus_dir / name #tsvファイルのパス

        #データモジュールの用意
        data_module_Livedoor = LivedoorDataModule(mode = "txt_classfication_by_title", corpus_path = corpus_path, batch_size = batch_size)

        #データセットの取得
        dataset = data_module_Livedoor.dataset

        #モデルの用意
        net, optimizer, loss_function = set_model(data_module_Livedoor.labels_dim)
        print(net)

        # dataloaderの作成
        train_dataloader = data_module_Livedoor.train_dataloader()
        val_dataloader = data_module_Livedoor.val_dataloader()
        test_dataloader = data_module_Livedoor.test_dataloader()

    # #モデルの学習
    trainer = Trainer(net, optimizer, loss_function, num_epochs)
    trainer.fit_model(train_dataloader, val_dataloader)

    # Test
    tester = Tester(net, loss_function)
    tester.test(test_dataloader)

    # Testデータで分類結果出力
    out_prediction = Out_prediction(net)
    out_prediction.out(test_dataloader)



#入力例
#python main.py --use_data Livedoor --batch_size 10 --max_epoch 100
