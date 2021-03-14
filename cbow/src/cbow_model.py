import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
torch.manual_seed(0)

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--vocab_size', dest='vocab_size', help='Look window size', default=10, type=int)
    parser.add_argument('--emb_dim', dest='emb_dim', help='Num of Embedding layer node', default=5, type=int)
    args = parser.parse_args()
    return args

#CBOWのモデル
class CBOW_model(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        output_dim = vocab_size #出力層の次元数は, 入力層の次元数と同じ
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)  #vocab_sizeはボキャブラリのサイズ. 入力サイズでなくてよい.
        torch.nn.init.xavier_uniform(self.embedding_layer.weight) #重みの初期化
        self.linear_layer_embedding2output = nn.Linear(embedding_dim, output_dim) #中間層 → 出力層
        torch.nn.init.xavier_uniform(self.linear_layer_embedding2output.weight) #重みの初期化


    def forward(self, inputs_word_ID):
        # print(self.embedding_layer(inputs_word_ID))
        embeds = self.embedding_layer(inputs_word_ID).sum(dim = 1)
        # print(embeds)

        out_x = self.linear_layer_embedding2output(embeds) #closs entropy lossを使う場合
        # out_x = F.softmax(self.linear_layer_embedding2output(embeds)) # nll lossを使う場合. softmaxを掛けているので, 結局closs entropyと同じ
        return out_x




if __name__ == "__main__":
    args = parse_args()
    model = CBOW_model(args.vocab_size, args.emb_dim)
    print(model)

    #出力形式等の確認用
    tmp_tensor = torch.tensor([1, 2, 3, 4])
    out = model.forward(tmp_tensor)
    print(out)
