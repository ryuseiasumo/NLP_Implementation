import torch
import numpy as np


list_x = [[1,0,1,1],[0,1,2,3],[0,2,0,2]]

np_x = np.array(list_x)

tensor_x = torch.tensor(list_x)


print(list_x)
print(np_x)
print(tensor_x)

print("転値")
print(torch.t(tensor_x)) #転値

print("sum")
print(tensor_x.sum(dim = 0)) #列方向でたされる
print(tensor_x.sum(dim = 1)) #行方向でたされる

print("view")
print(tensor_x.view(-1,1)) #行か列のどちらかを固定、どちらかを-1に設定して、任意の形に変更させる
print(tensor_x.view(-1,2))
print(tensor_x.view(1,-1))
print(tensor_x.view(2,-1))
print(tensor_x.view(4,3))
print(tensor_x.sum().view(1,-1))

print("reshape(viewとほぼ同じ)")
print(torch.reshape(tensor_x,(4,3)))


tensor_y = torch.tensor(list_x)
print(tensor_x)
print(tensor_y)
print(torch.cat([tensor_x,tensor_y]))

    # def forward(self, inputs_word_ID):
    #     # print(self.embedding_layer(inputs_word_ID))
    #     embeds = self.embedding_layer(inputs_word_ID).sum(dim = 1)
    #     # embeds = self.embedding_layer(inputs_word_ID).sum(dim = 0).view(1,-1)
    #     print(embeds)

if str(type(list_x)) == "<class 'list'>":
    print(type(list_x))
    print("True")


a =  (torch.tensor([21798, 28599, 2867, 3]), torch.tensor(28599))
print(a[1])
print(a[1].view(1))
print(a[0].view(1,4,1)) #1を入れると次元数を上げることができる
c = torch.cat([a[0],a[1].view(1)])
print(c)
print(torch.max(c).tolist())


a_list =  [(torch.tensor([21798, 28599, 2867, 3]), torch.tensor(28599)),(torch.tensor([21798, 30000, 2867, 3]), torch.tensor(28599))]

max_list = [torch.max(torch.cat([i[0],i[1].view(1)])) for i in a_list]
print(max_list)
