from torch import nn
import torch
import numpy as np
from einops import repeat, rearrange


# a = np.arange(10)
# print(a)
# np.random.shuffle(a)
# print(a)
# b = np.argsort(a)
# print(b)

# 下面代码出现问题:问题部分result_a[1][0][3] = In[index_a[1][0][3]][0][3] 此时index_a[1][0][3]=2 而In的第0个维度最大为1
# 此时必须满足index_a中的值小于batch=2
a = torch.arange(24).reshape(2,3,4) # [batch=2,seq=3,embedding_size]
index_a = torch.tensor([[1,0],[0,1]]) # [2,3]
index_a = repeat(index_a, 't b -> t b e',e=4) # [2,3,4]
# 此时的result_a[0][0][0] = a[index_a[0][0][0]][0][0] = a[1][0][0]这个是不对的，
# 此时result_a的第0个批量的第0个patch来自于输入的第0个批量的第0个patch
result_a = torch.gather(a,dim=0,index=index_a)


# b = a.permute(1,0,2) # [seq=3,batch=2,embedding_size]
# index_b = torch.tensor([[1,0],[1,0],[0,1]]) # [3,2]
# index_b = repeat(index_b, 'b t -> b t e',e=4)
# result_b = torch.gather(b,dim=0,index=index_b)
#
# print(result_a == result_b.permute(1,0,2))

# a = torch.tensor([[[1,2,3]],[[3,4,5]]])
# print(a.shape)
# b = a.expand(-1,2,-1)
# print(b.shape)

# import math
# lr_func = lambda epoch: min((epoch + 1) / (200 + 1e-8),
#                             0.5 * (math.cos(epoch / 2000 * math.pi) + 1))
#
# from torchvision.utils import save_image
# import torchvision
#
# trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
#                                         torchvision.transforms.Normalize(0.5, 0.5)])
# test_dataset = torchvision.datasets.CIFAR10('D:\\python\\pytorch作业\\all_data\\cifar10\\data',
#                                             train=False, transform=trans, download=False)
# val_img = torch.stack([test_dataset[i][0] for i in range(16)],dim=0)
#
# save_image(val_img,f'images/{1}.png',nrow=2,normalize=True)

# a = torch.arange(120).reshape(2,3,4,5)
# b = rearrange(a,'b c w h -> b (w h) c')
# d = a.permute(0,2,3,1)
# d = d.reshape(2,-1,3)
# print(b == d)
#
# c = rearrange(a,'b c w h -> b (h w) c')
# e = a.permute(0,3,2,1)
# e = e.reshape(2,-1,3)
# print(c == e)

