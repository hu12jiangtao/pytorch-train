from torch import nn
from d2l import torch as d2l
import torch

#构建一个vgg块
def vgg_block(in_channal,out_channal,channal_num):
    block_list=[]
    for i in range(channal_num):
        block_list.append(nn.Conv2d(in_channal,out_channal,kernel_size=3,padding=1))
        block_list.append(nn.ReLU())
        in_channal=out_channal
    block_list.append(nn.MaxPool2d(stride=2,kernel_size=2))
    return nn.Sequential(*block_list)

def vgg_net(net_block):
    vgg_list=[]
    in_channal=1
    for channal_num,out_channal in net_block:
        vgg_list.append(vgg_block(in_channal,out_channal,channal_num))
        in_channal=out_channal
    net=nn.Sequential(*vgg_list)
    net.add_module('Flatten',nn.Flatten())
    net.add_module('Linear1',nn.Linear(7*7*out_channal,4096))
    net.add_module('acticate1',nn.ReLU())
    net.add_module('drop_out1',nn.Dropout(0.5))
    net.add_module('Linear2',nn.Linear(4096,4096))
    net.add_module('acticate2',nn.ReLU())
    net.add_module('drop_out2',nn.Dropout(0.5))
    net.add_module('Linear3',nn.Linear(4096,10))
    return net

net_block=((1,64),(1,128),(2,256),(2,512),(2,512))
small_net_block=((i,j//8) for i,j in net_block)
net=vgg_net(small_net_block)
X=torch.randn((1,1,224,224),dtype=torch.float32)
for layer in net:
    X=layer(X)
    print(layer.__class__.__name__,'output shape:',X.shape)