import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import trange, tqdm

torch.manual_seed(0)

#評価
def test(test_loader, net): #net = cbow_model.CBOW_model
    net.eval()
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()

    running_loss = 0
    correct = 0
    total = 0
    l = len(test_loader)
    for i, (input_data, labels) in tqdm(enumerate(test_loader),total = l):
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
        # print(outputs, labels)

        loss = loss_function(outputs, labels)
        running_loss += loss.data

        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels.data).sum().item()
        total += labels.size(0)

    val_loss = running_loss / len(test_loader)
    val_acc = correct / total

    return val_loss, val_acc, correct, total
