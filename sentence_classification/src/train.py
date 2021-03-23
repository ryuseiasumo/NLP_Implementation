import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from tqdm import trange, tqdm

#シード値
torch.manual_seed(0)

class Trainer:
    def __init__(self, net, optimizer, loss_function, num_epochs):
        super().__init__()
        self.net = net
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.num_epochs = num_epochs

    #訓練用
    def train(self, train_loader):
        self.net.train()
        running_loss, correct, total = 0, 0, 0
        l = len(train_loader)

        for i, (input_data, labels) in tqdm(enumerate(train_loader),total = l):
            # print(i, (input_data, labels))
            if torch.cuda.is_available():
                input_data = Variable(input_data.cuda(), volatile=True)
                labels = Variable(labels.cuda(), volatile=True)
            else:
                input_data = Variable(input_data, volatile=True)
                labels = Variable(labels, volatile=True)

            self.optimizer.zero_grad()
            outputs = self.net(input_data)

            loss = self.loss_function(outputs, labels)
            running_loss += loss.data
            loss.backward()
            self.optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels.data).sum().item()
            total += labels.size(0)

        train_loss = running_loss / l
        train_acc = correct / total

        return train_loss, train_acc , correct, total


    #検証用
    def valid(self,val_loader):
        self.net.eval()
        running_loss, correct, total = 0, 0, 0
        l = len(val_loader)

        for i, (input_data, labels) in tqdm(enumerate(val_loader),total = l):
            if torch.cuda.is_available():
                input_data = Variable(input_data.cuda(), volatile=True)
                labels = Variable(labels.cuda(), volatile=True)
            else:
                input_data = Variable(input_data, volatile=True)
                labels = Variable(labels, volatile=True)

            outputs = self.net(input_data)
            loss = self.loss_function(outputs, labels)
            running_loss += loss.data

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels.data).sum().item()
            total += labels.size(0)

        val_loss = running_loss / l
        val_acc = correct / total

        return val_loss, val_acc, correct, total


    def fit_model(self, train_dataloader, val_dataloader): #実際にモデルの学習を行う関数(early_stoppingあり)
        early_stopping = Early_Stopping(patience=5, DoOrNot=1)

        loss_list = []
        val_loss_list = []
        val_acc_list = []
        val_acc_end=0.0

        # Training
        for epoch in trange(self.num_epochs):
            train_loss, train_acc , train_correct, train_total = self.train(train_dataloader)
            val_loss, val_acc, val_correct, val_total  = self.valid(val_dataloader)
            val_acc_end=val_acc #追記

            print('epoch %d, train_loss: %.4f validation_loss: %.4f min_loss: %.4f train_accuracy: %.4f(%d/%d) val_accuracy: %.4f(%d/%d)'
            % (epoch, train_loss, val_loss, early_stopping.loss, train_acc, train_correct, train_total, val_acc_end, val_correct, val_total))

            if early_stopping.validate(val_loss):
                break

            #logging
            loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)

        print('validation_accuracy: %.4f' % val_acc_end)
        print('Finished training')



#Early stoppingクラス
class Early_Stopping():
    def __init__(self, patience=0, DoOrNot = 0):
        self.step = 0
        self.loss = float('inf')  #誤差の初期値は∞
        self.patience = patience  #過去どれだけのエポック数までの誤差をみるか
        self.DoOrNot = DoOrNot

    def validate(self, _loss):
        if (self.loss < _loss):  #これまでのエポックに比べ、誤差が増えた
            self.step += 1
            if self.step > self.patience: #誤差が、patience回以上増え続けた
                if self.DoOrNot: #EarlyStoppingするかどうか
                    print("Early Stopping")
                    return True

        else:
            self.step = 0 #0に戻す
            self.loss = _loss #今回のエポックが誤差が最小なので、更新。
            return False
