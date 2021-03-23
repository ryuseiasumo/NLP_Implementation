import random
from pathlib import Path
SWD = Path(__file__).resolve().parent

# sys.path.append(os.path.abspath(SWD / "../../.."))

sample_text = """We are about to study the idea of a computational process.
            Computational processes are abstract beings that inhabit computers .
            As they evolve , processes manipulate other abstract things called data .
            The evolution of a process is directed by a pattern of rules
            called a program . People create programs to direct processes . In effect ,
            we conjure the spirits of the computer with our spells .""".split()

# print(set(sample_text))

data = []
window_size = 2

for i in range(window_size, len(sample_text) - window_size):
    context = list(sample_text[i + j] for j in range(-window_size, window_size+1) if j != 0)
    target = sample_text[i]
    data.append((context, target))

print(data)

random.shuffle(sample_text)

# print(sample_text)



#livedoorニュースのデータセット作成の
from torch.utils.data import Dataset
import random


class TsvDataset(Dataset):
    def __init__(self, corpus_path: Path, transform=lambda x: x):
        super().__init__()
        self.dataset = []

        with corpus_path.open() as f:
            self.dataset.extend([transform(line.strip().split("\t")) for line in f])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]



corpus_dir = Path("corpus/livedoor")
name = "train.tsv"

corpus_path = SWD /corpus_dir / name
print(corpus_path)

livedoor = TsvDataset(corpus_path)
print(livedoor.dataset)



from transformers import BertJapaneseTokenizer

# 日本語BERT用のtokenizerを宣言
tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")

text = "自然言語処理はとても楽しい。"

wakati_ids = tokenizer.encode(text, max_length=80, padding="max_length")
print(tokenizer.convert_ids_to_tokens(wakati_ids))
print(wakati_ids)


# wakati_ids_tensor = tokenizer.encode(text, return_tensors='pt') #tensor型による出力
# print(tokenizer.convert_ids_to_tokens(wakati_ids_tensor[0].tolist()))
# print(wakati_ids_tensor)
