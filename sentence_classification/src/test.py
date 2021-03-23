import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from tqdm import trange, tqdm

#シード値
torch.manual_seed(0)

class Tester:
    def __init__(self, net, loss_function):
        super().__init__()
        self.net = net
        self.loss_function = loss_function

    def test(self, test_loader):
        # Test
        self.net.eval()

        running_loss, test_correct, test_total = 0, 0, 0
        l = len(test_loader)

        for i, (input_data, labels) in tqdm(enumerate(test_loader),total = l):
            if torch.cuda.is_available():
                input_data = Variable(input_data.cuda(), volatile=True)
                labels = Variable(labels.cuda(), volatile=True)
            else:
                input_data = Variable(input_data, volatile=True)
                labels = Variable(labels, volatile=True)

            outputs = self.net(input_data)
            # print(outputs, labels)

            loss = self.loss_function(outputs, labels)
            running_loss += loss.data

            _, predicted = torch.max(outputs.data, 1)
            test_correct += (predicted == labels.data).sum().item()
            test_total += labels.size(0)

        test_loss = running_loss / l
        test_acc = test_correct / test_total

        print('test_loss: %.4f test_accuracy: %.4f(%d/%d)'
                % (test_loss, test_acc, test_correct, test_total))

        # return test_loss, test_acc, test_correct, test_total
