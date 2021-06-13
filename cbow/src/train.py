import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from tqdm import trange, tqdm

torch.manual_seed(0)

def train(train_loader, net): #net = cbow_model.CBOW_model
    net.train()

    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()
    # loss_function = nn.NLLLoss() #softmaxを使う場合. CrossEntropyLoss()と同じになる.

    running_loss = 0
    l = len(train_loader)
    correct = 0
    total = 0

    for i, (input_data, labels) in tqdm(enumerate(train_loader),total = l):
        if torch.cuda.is_available():
            with torch.no_grad():
                input_data = Variable(input_data.cuda())
                labels = Variable(labels.cuda())
        else:
            with torch.no_grad():
                input_data = Variable(input_data)
                labels = Variable(labels)

        optimizer.zero_grad()
        outputs = net(input_data)

        loss = loss_function(outputs, labels)
        running_loss += loss.data

        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels.data).sum().item()
        total += labels.size(0)


    train_loss = running_loss / len(train_loader)
    train_acc = correct / total

    return train_loss, train_acc , correct, total


#評価
def valid(val_loader, net):
    net.eval()
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()
    running_loss = 0
    correct = 0
    total = 0
    l = len(val_loader)
    for i, (input_data, labels) in tqdm(enumerate(val_loader),total = l):
        if torch.cuda.is_available():
            with torch.no_grad():
                input_data = Variable(input_data.cuda())
                outputs = net(input_data)
                labels = Variable(labels.cuda())
        else:
            with torch.no_grad():
                input_data = Variable(input_data)
                outputs = net(input_data)
                labels = Variable(labels)

        loss = loss_function(outputs, labels)
        running_loss += loss.data

        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels.data).sum().item()
        total += labels.size(0)

    val_loss = running_loss / len(val_loader)
    val_acc = correct / total

    return val_loss, val_acc, correct, total




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
