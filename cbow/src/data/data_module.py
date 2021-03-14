import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from . import datasets
from pathlib import Path
import sys, os

SWD = Path(__file__).resolve().parent

sys.path.append(os.path.abspath(SWD / "../../.."))

# from make_expanded import generate_tokenizer


class SampleDataModule:
    def __init__(self, mode, add_word_path: Path, batch_size, window_size = 1):
        self.batch_size = batch_size
        self.tokenizer = None
        self.dataset = datasets.SampleDataset(self.transform_word2idx, window_size)
        self.dataset_length = self.dataset.len()

    def transform_word2idx(list_x, dict_word2idx):
        # print(list_x)
        list_ids = [ (torch.tensor([dict_word2idx[key] for key in context_datas], dtype=torch.long), torch.tensor(dict_word2idx[target_data])) for (context_datas, target_data) in list_x ]

        return list_ids

    # def transform_word2idx(self, x, dict_word2idx):
    #     context_datas, target_data = x

    #     return (
    #         torch.tensor(
    #             [dict_word2idx[key] for key in context_datas]
    #         ),
    #         torch.tensor(dict_word2idx[target_data]),
    #     )

    # def train_dataloader(self):
    #     return DataLoader(
    #         self.dataset.data,
    #         batch_size=self.batch_size,
    #         shuffle=True,
    #     )

    # def dev_dataloader(self): #今回はearlystoppingしない
    #     return DataLoader(
    #         self.dataset[
    #             int(self.dataset_length / 10 * 8) : int(self.dataset_length / 10 * 9)
    #         ],
    #         batch_size=self.batch_size,
    #     )

    # def test_dataloader(self):
    #     return DataLoader(
    #         self.dataset.test_data,
    #         batch_size=self.batch_size,
    #     )

    # def kfold_dataloader(self, n_splits: int):
    #     for train_index, dev_index, test_index in self.kfold_dataloader_index(n_splits):
    #         train_dataset = Subset(self.dataset, train_index)
    #         dev_dataset = Subset(self.dataset, dev_index)
    #         test_dataset = Subset(self.dataset, test_index)

    #         train_dataloader = DataLoader(
    #             train_dataset, batch_size=self.batch_size, shuffle=True
    #         )
    #         dev_dataloader = DataLoader(dev_dataset, batch_size=self.batch_size)
    #         test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size)
    #         yield train_dataloader, dev_dataloader, test_dataloader

    # def kfold_dataloader_index(self, n_splits: int):
    #     n_splits = int(n_splits / 2)
    #     kf = KFold(n_splits=n_splits)
    #     for train_index, dev_test_index in kf.split(self.dataset.dataset):
    #         border = int(len(dev_test_index) / 2)
    #         dev_index, test_index = dev_test_index[:border], dev_test_index[border:]

    #         yield train_index, dev_index, test_index

    #         dev_index, test_index = test_index, dev_index

    #         yield train_index, dev_index, test_index






# class SampleDataModule:
#     def __init__(self, mode, add_word_path: Path, batch_size):
#         self.labels = None
#         self.batch_size = batch_size

#         self.label2id = None
#         self.id2label = None
#         self.tokenizer = None

#         self.dataset = datasets.SampleDataset()
#         # self.dataset = datasets.SampleDataset(self.transform_title)

#         self.dataset_length = len(self.dataset)

#     # def transform_title(self, x):
#     #     title, _, label = x
#     #     return (
#     #         torch.tensor(
#     #             self.tokenizer(title, max_length=80, padding="max_length")["input_ids"]
#     #         ),
#     #         torch.tensor(self.label2id[label]),
#     #     )

#     def train_dataloader(self):
#         return DataLoader(
#             self.dataset[: int(self.dataset_length / 10 * 8)],
#             batch_size=self.batch_size,
#             shuffle=True,
#         )

#     def dev_dataloader(self):
#         return DataLoader(
#             self.dataset[
#                 int(self.dataset_length / 10 * 8) : int(self.dataset_length / 10 * 9)
#             ],
#             batch_size=self.batch_size,
#         )

#     def test_dataloader(self):
#         return DataLoader(
#             self.dataset[int(self.dataset_length / 10 * 9) :],
#             batch_size=self.batch_size,
#         )

#     def kfold_dataloader(self, n_splits: int):
#         for train_index, dev_index, test_index in self.kfold_dataloader_index(n_splits):
#             train_dataset = Subset(self.dataset, train_index)
#             dev_dataset = Subset(self.dataset, dev_index)
#             test_dataset = Subset(self.dataset, test_index)

#             train_dataloader = DataLoader(
#                 train_dataset, batch_size=self.batch_size, shuffle=True
#             )
#             dev_dataloader = DataLoader(dev_dataset, batch_size=self.batch_size)
#             test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size)
#             yield train_dataloader, dev_dataloader, test_dataloader

#     def kfold_dataloader_index(self, n_splits: int):
#         n_splits = int(n_splits / 2)
#         kf = KFold(n_splits=n_splits)
#         for train_index, dev_test_index in kf.split(self.dataset.dataset):
#             border = int(len(dev_test_index) / 2)
#             dev_index, test_index = dev_test_index[:border], dev_test_index[border:]

#             yield train_index, dev_index, test_index

#             dev_index, test_index = test_index, dev_index

#             yield train_index, dev_index, test_index
