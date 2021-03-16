import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from . import datasets
from pathlib import Path
import sys, os

from transformers import BertJapaneseTokenizer


SWD = Path(__file__).resolve().parent

sys.path.append(os.path.abspath(SWD / "../../.."))

# from make_expanded import generate_tokenizer


class SampleDataModule:
    def __init__(self, mode, corpus_path: Path, batch_size, window_size = 1):
        self.batch_size = batch_size
        self.tokenizer = None
        self.dataset = datasets.SampleDataset(self.transform_word2idx, window_size)
        self.dataset_length = self.dataset.len()

    def transform_word2idx(list_x, dict_word2idx):
        # print(list_x)
        list_ids = [ (torch.tensor([dict_word2idx[key] for key in context_datas], dtype=torch.long), torch.tensor(dict_word2idx[target_data])) for (context_datas, target_data) in list_x ]

        return list_ids



class LivedoorDataModule:
    def __init__(self, mode, corpus_path: Path, batch_size, window_size = 2):
        #txt分類用のラベル
        self.labels = [
            "it-life-hack",
            "livedoor-homme",
            "peachy",
            "smax",
            "topic-news",
            "dokujo-tsushin",
            "kaden-channel",
            "movie-enter",
            "sports-watch",
        ]
        self.batch_size = batch_size
        self.corpus_path = corpus_path
        self.window_size = window_size
        self.label2id = {label: _id for _id, label in enumerate(self.labels)}
        self.id2label = {value: key for key, value in self.label2id.items()}
        self.tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
        # self.train, self.val, self.test = None, None, None

        if mode == "cbow":
            self.dataset = datasets.TsvDataset(self.corpus_path, self.transform_for_cbow, self.get_vocab_size, window_size = self.window_size)
        elif mode == "txt_classfication_by_title":
            self.dataset = datasets.TsvDataset(self.corpus_path, self.transform_for_txt_classfication_by_title)

        self.dataset_length = len(self.dataset)

    def get_vocab_size(self, tmp_max_idx, data): #語彙数を取得する関数
        #dataの形によって場合わけ
        if str(type(data)) == "<class 'tuple'>": #tuple型の場合
            data_cat = torch.cat([data[0], data[1].view(1)])
            max_idx_of_this_data = torch.max(data_cat).tolist()

        elif str(type(data)) == "<class 'list'>": #list型の場合
            data_max_list = [torch.max(torch.cat([data_i[0],data_i[1].view(1)])).tolist() for data_i in data]
            max_idx_of_this_data = max(data_max_list)

        max_idx_of_this_data += 1 #max = 3なら, [0,1,2,3]で語彙数は4であるため.

        if tmp_max_idx < max_idx_of_this_data:
            new_max_idx = max_idx_of_this_data
        else:
            new_max_idx = tmp_max_idx

        return new_max_idx


    def transform_for_cbow(self, x, window_size = 2): #cbow用
        _, contents, _ = x
        list_context_and_target = []

        #ある文を形態素解析(分かち書き)する
        wakati_ids = self.tokenizer.encode(contents, max_length=80, padding="max_length")

        #windowサイズに合わせて, contextとtargetを生成
        for i in range(window_size, len(wakati_ids) - window_size): #パディングしたのも含んでいる.
            context = list(wakati_ids[i + j] for j in range(-window_size, window_size+1) if j != 0)
            target = wakati_ids[i]
            list_context_and_target.append((torch.tensor(context), torch.tensor(target)))
        return list_context_and_target


    def transform_for_txt_classfication_by_title(self, x): #タイトルからテキスト分類
        title, _, label = x
        return (
            torch.tensor(
                self.tokenizer(title, max_length=80, padding="max_length")["input_ids"]
            ),
            torch.tensor(self.label2id[label]),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset[: int(self.dataset_length / 10 * 8)], #要修正？分けられている場合は、self.trainにする？
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset[
                int(self.dataset_length / 10 * 8) : int(self.dataset_length / 10 * 9)
            ],
            batch_size=self.batch_size,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset[int(self.dataset_length / 10 * 9) :],
            batch_size=self.batch_size,
        )
