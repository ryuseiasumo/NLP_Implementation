import torch
import torch.nn as nn
import argparse

import torch.nn.functional as F
from torch import optim

from tqdm import trange, tqdm

torch.manual_seed(0)

from src import cbow_model

from src.data import data_module
from src.data import datasets

from src.train import train
from src.train import valid #追記
from src.train import Early_Stopping
from src.test import test

from pathlib import Path
SWD = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--use_data', dest='use_data', help='which data use(Sample, Livedoor)', default="Sample", type=str)
    parser.add_argument('--window_size', dest='window_size', help='Look window size', default=1, type=int)
    parser.add_argument('--vocab_size', dest='vocab_size', help='Look window size', default=10, type=int)
    parser.add_argument('--emb_dim', dest='emb_dim', help='Num of Embedding layer node', default=5, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size', default=10, type=int)
    parser.add_argument('--max_epoch', dest='max_epoch', help='Num of max epoch', default=100, type=int)

    args = parser.parse_args()
    return args

args = parse_args()
num_epochs = args.max_epoch
batch_size = args.batch_size


def set_model(vocab_size, emb_dim):
    net = cbow_model.CBOW_model(vocab_size, emb_dim)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=0.001)
    # GPUを使うか判定
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    net.to(device)
    return net, loss_function, optimizer


if __name__ == "__main__":
    if args.use_data == "Sample":
        #サンプルデータセットによる挙動確認

        #データセットの取得
        transform = data_module.SampleDataModule.transform_word2idx
        dataset = datasets.SampleDataset(transform=transform, window_size= args.window_size)

        # print(dataset.data)
        # print(dataset.test_data)
        # print(dataset.word_to_idx)

        #モデルの用意
        net, loss_function, optimizer = set_model(args.vocab_size, args.emb_dim)
        print(net)

        # dataloaderの作成
        train_dataloader = torch.utils.data.DataLoader(dataset.data, batch_size= batch_size, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(dataset.test_data, batch_size= batch_size, shuffle=True)

        # early_stopping = Early_Stopping(patience=5, DoOrNot=1)

        loss_list = []

        # Training
        for epoch in trange(num_epochs):
            train_loss, train_acc , train_correct, train_total = train(train_dataloader, net)

            print('epoch %d, train_loss: %.4f train_accuracy: %.4f(%d/%d)'
                % (epoch, train_loss, train_acc, train_correct, train_total))

            #logging
            loss_list.append(train_loss)
            # val_loss_list.append(val_loss)
            # val_acc_list.append(val_acc)

        # print('validation_accuracy: %.4f' % val_acc_end)
        print('Finished training')


    elif args.use_data == "Livedoor":
        #livedoorニュースによる学習
        corpus_dir = Path("corpus/livedoor")
        name = "all.tsv"
        corpus_path = SWD /corpus_dir / name #tsvファイルのパス

        #データモジュールの用意
        data_module_Livedoor = data_module.LivedoorDataModule(mode = "cbow", corpus_path = corpus_path, batch_size = batch_size, window_size= args.window_size)

        #データセットの取得
        dataset = data_module_Livedoor.dataset

        #モデルの用意
        net, loss_function, optimizer = set_model(dataset.vocab_size, args.emb_dim)
        print(net)

        # dataloaderの作成
        train_dataloader = data_module_Livedoor.train_dataloader()
        val_dataloader = data_module_Livedoor.val_dataloader() #追記
        test_dataloader = data_module_Livedoor.test_dataloader()

        early_stopping = Early_Stopping(patience=5, DoOrNot=1) #追記

        loss_list = []
        val_loss_list = [] #追記
        val_acc_list = [] #追記
        val_acc_end=0.0 #追記

        # Training
        for epoch in trange(num_epochs):
            train_loss, train_acc , train_correct, train_total = train(train_dataloader, net)
            # val_loss, val_acc = valid(val_dataloader) #追記
            val_loss, val_acc, val_correct, val_total  = valid(val_dataloader, net) #追記
            val_acc_end=val_acc #追記

            print('epoch %d, train_loss: %.4f validation_loss: %.4f min_loss: %.4f train_accuracy: %.4f(%d/%d) val_accuracy: %.4f(%d/%d)'
            % (epoch, train_loss, val_loss, early_stopping.loss, train_acc, train_correct, train_total, val_acc_end, val_correct, val_total))

            if early_stopping.validate(val_loss):
                break
            # print('epoch %d, train_loss: %.4f train_accuracy: %.4f(%d/%d)'
            #     % (epoch, train_loss, train_acc, train_correct, train_total))

            #logging
            loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)

        print('validation_accuracy: %.4f' % val_acc_end)
        print('Finished training')


    # Test
    test_loss, test_acc , test_correct, test_total = test(test_dataloader, net)
    print('test_loss: %.4f test_accuracy: %.4f(%d/%d)'
            % (test_loss, test_acc, test_correct, test_total))



#入力例
#python main.py --use_data Sample --window_size 2 --vocab_size 50 --emb_dim 15 --batch_size 10 --max_epoch 100
