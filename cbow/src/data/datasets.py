from torch.utils.data import Dataset
from pathlib import Path
import random

SWD = Path(__file__).resolve().parent

#Cbow用のサンプルデータセット
class SampleDataset(Dataset):
    def __init__(self, transform=None, window_size = 1):
        super().__init__()
        sample_text = """We are about to study the idea of a computational process.
            Computational processes are abstract beings that inhabit computers .
            As they evolve , processes manipulate other abstract things called data .
            The evolution of a process is directed by a pattern of rules
            called a program . People create programs to direct processes . In effect ,
            we conjure the spirits of the computer with our spells .""".split()

        self.vocab = set(sample_text) #ボキャブラリ
        self.vocab_size = len(self.vocab) #語彙数
        self.word_to_idx = {word:idx for idx, word in enumerate(self.vocab)} #単語→id
        self.idx_to_word = {idx:word for idx, word in enumerate(self.vocab)} #id→単語

        self.data = [] #データセット(入力コンテキストデータ, 教師データ). SampleDatasetではこれをすべてTrainで使う
        for i in range(window_size, len(sample_text) - window_size):
            context = list(sample_text[i + j] for j in range(-window_size, window_size+1) if j != 0)
            target = sample_text[i]
            self.data.append((context, target))

        self.test_data = (['People','create','to', 'direct'], 'programs') #testデータ. 本来はdataからランダムに選ぶがSampleDatasetは挙動確認が目的のため, このように与えている.

        self.transform = transform
        if self.transform:
            self.data = self.transform(self.data, self.word_to_idx)
            self.test_data = self.transform([self.test_data], self.word_to_idx)
        random.shuffle(self.data)
        # print(self.data)
        # print(self.test_data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        # if self.transform:
        #     data = self.transform(self.data, self.word_to_idx)

        return self.data[index]
