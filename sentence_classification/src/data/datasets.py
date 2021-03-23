from torch.utils.data import Dataset
from pathlib import Path
import random
import numpy as np
import torch #動作確認用

SWD = Path(__file__).resolve().parent

#Livedoor newsデータセット
class TsvDataset(Dataset):
    def __init__(self, corpus_path: Path, transform=None): #transformが特になければ、ただそのまま返す関数(lambda x:x)とする.
        super().__init__()
        self.dataset = []

        with corpus_path.open() as f_corpus:
            if transform == None: #transformなし.　コーパスを[タイトル, 内容, ラベル]のリストにして返す.
                transform = lambda x: x
                self.dataset.extend([transform(line.strip().split("\t")) for line in f_corpus ]) #extendメソッド：他のリストの要素を追加

            else: #transformあり
                for line in f_corpus:
                    #　[タイトル, 内容, ラベル]の順で分割, 分かち書き, id化などのtransform内で定義された処理を実行
                    data = transform(line.strip().split("\t"))
                    # print(type(data))
                    if str(type(data)) == "<class 'list'>": #Cbowなどは, transform内でデータと教師信号を作成し, リスト化するので, リストとして返される. 故に他と同様の遣り方だと二次元配列になってしまう([[(tensor, tensor),(tensor, tensor),...(tensor, tensor)],[(tensor, tensor),(tensor, tensor),..],...[(tensor, tensor),(tensor, tensor),...]])
                        self.dataset += data

                    else: #txt分類などは, transform以前に教師信号がlabelとして得ており, データもすでに形になっている.[(tensor, tensor),(tensor, tensor),...,(tensor, tensor)]
                        self.dataset.append(data)

            random.shuffle(self.dataset)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]






# ------- 以下, 動作確認用の関数 ---------
def transform_tmp(x):
    title, _, label = x
    return (torch.tensor([1,2,3]), torch.tensor([1]))

from transformers import BertJapaneseTokenizer
def transform_tmp_2(x, window_size = 2):
    _, contents, _ = x
    list_context_and_target = []

    #ある文を形態素解析(分かち書き)する
    tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
    wakati_ids = tokenizer.encode(contents, max_length=80, padding="max_length")

    #windowサイズに合わせて, contextとtargetを生成
    for i in range(window_size, len(wakati_ids) - window_size): #パディングしたのも含んでいる.
        context = list(wakati_ids[i + j] for j in range(-window_size, window_size+1) if j != 0)
        target = wakati_ids[i]
        list_context_and_target.append((torch.tensor(context), torch.tensor(target)))

    return list_context_and_target

def get_vocab_size(tmp_max_idx, data): #語彙数を取得する関数
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

# ------------------------------



if __name__ == "__main__":
    #動作確認用
    corpus_dir = Path("corpus/livedoor")
    name = "train.tsv"

    # corpus_path = SWD /corpus_dir / name

    corpus_path = SWD / "../../" /corpus_dir / name
    print(corpus_path)

    livedoor = TsvDataset(corpus_path)
    print(livedoor.dataset[0])

    np_livedoor = np.array(livedoor)
    # print(np_livedoor[:,2])
    print(len(np_livedoor[np_livedoor[:,2] == "it-life-hack"]))
    print(np_livedoor.shape)

    livedoor = TsvDataset(corpus_path, transform_tmp, get_vocab_size)
    print(livedoor.dataset[0])
    print(len(livedoor.dataset))
    print(livedoor.vocab_size)

    livedoor = TsvDataset(corpus_path, transform_tmp_2, get_vocab_size)
    print(livedoor.dataset[0])
    print(livedoor.vocab_size)
    print(len(livedoor.dataset))
