import torch
from torch import nn
from transformers import BertModel
# from transformers.tokenization_bert_japanese import BertJapaneseTokenizer

#Bertを用いたテキスト分類モデル
class SentenceClassification(nn.Module):
    def __init__(self, label_dim: int):
        super().__init__()
        BERT_Emb_dim = 768
        self.label_dim = label_dim
        self.bert = BertModel.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking") #日本語版BERTの事前学習モデル
        self.linear_layer_classfication = nn.Linear(BERT_Emb_dim, label_dim) #タスク(テキスト分類)用全結合層
        torch.nn.init.xavier_uniform(self.linear_layer_classfication.weight) #重みの初期化
        # self.criterion = nn.CrossEntropyLoss()

    @staticmethod #selfに影響しない
    def _make_attention_mask(batch): #デコーダにおいて, パディングの部分のアテンションを0にする
        return (batch != 0).float()

    def forward(self, batch):
        attention_mask = self._make_attention_mask(batch) #アテンションマスク
        cls = self.bert(batch, attention_mask=attention_mask)[1] #clsは文についての分散表現で(1x768)次元
        pred_labels = self.linear_layer_classfication(cls) #clsからテキスト分類する全結合層
        # pred_labels = self.linear_layer_classfication(cls).view(-1, self.label_dim) #clsからテキスト分類する全結合層
        # print(pred_labels.argmax(dim=1))

        return pred_labels

    def out_prediction(self, batch): #分類結果を出力する関数
        pred_labels = self.forward(batch)
        # print(pred_labels.argmax(dim=1))
        return pred_labels.argmax(dim=1)
