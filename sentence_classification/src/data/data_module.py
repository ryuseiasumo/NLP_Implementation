import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from .datasets import TsvDataset #__init__.pyでfrom .datasets import *としているため
from pathlib import Path
import sys, os

from transformers import BertJapaneseTokenizer


SWD = Path(__file__).resolve().parent

sys.path.append(os.path.abspath(SWD / "../../.."))

# from make_expanded import generate_tokenizer

class LivedoorDataModule:
    def __init__(self, mode, corpus_path: Path, batch_size):
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
        self.label2id = {label: _id for _id, label in enumerate(self.labels)}
        self.id2label = {value: key for key, value in self.label2id.items()}
        self.tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
        # self.train, self.val, self.test = None, None, None

        if mode == "txt_classfication_by_title":
            self.dataset = TsvDataset(self.corpus_path, self.transform_for_txt_classfication_by_title)

        self.dataset_length = len(self.dataset)
        self.labels_dim = len(self.labels)

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
            self.dataset[: int(self.dataset_length / 10 * 8)
            ],
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
