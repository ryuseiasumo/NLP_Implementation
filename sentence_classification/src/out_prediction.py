import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from tqdm import trange, tqdm

#シード値
torch.manual_seed(0)

class Out_prediction: #予想クラスを出力する関数
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.pred_labels_list = []
        self.gt_labels_list = []

    def out(self, data_loader):
        # Test
        self.net.eval()
        l = len(data_loader)

        for i, (input_data, labels) in tqdm(enumerate(data_loader),total = l):
            if torch.cuda.is_available():
                with torch.no_grad():
                    input_data = Variable(input_data.cuda())
                    pred_labels = self.net.out_prediction(input_data)
                    labels = Variable(labels.cuda())

            else:
                with torch.no_grad():
                    input_data = Variable(input_data)
                    pred_labels = self.net.out_prediction(input_data)
                    labels = Variable(labels)

            self.pred_labels_list += pred_labels.tolist()
            self.gt_labels_list += labels.tolist()

        print("---予想クラス---\n",self.pred_labels_list)
        print("---正解クラス---\n",self.gt_labels_list)
