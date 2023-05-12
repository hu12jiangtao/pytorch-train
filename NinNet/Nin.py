# Nin网络是对Alex net的一种改进，我们可以注意到将卷积神经网络Flatten后连接的全连接层，需要训练的参数个数有5*5*256*4096，参数的数量太多了
# Nin网络利用1*1的卷积神经网络来替换全连接层，同时也使用了块的概念，一个块中包含了一个卷积加上两个1*1的卷积（1*1的卷积的输入通道，输出通道数都等于你最后块的输出通道数）
# 使用1*1卷积层的参数个数也是输入的通道数*输出的通道数 等价于 全连接的输入神经元个数*输出神经元个数
# 例如96*27*27经过96*96*1*1的1*1卷积块得到96*27*27相当于样本个数为27*27，特征个数为96的输入；输出为特征个数为96，样本个数为27*27的全连接层
# 全局的平均池化层的作用是将figure_size进行了压缩
# 全局的平均池化层 优点：需要训练的参数个数变少了，提升了泛化性，使得你的精度变得更好，缺点有：收敛的速度变慢,作用是将一个面的数据做平均进行融合

from torch import nn
import torch
from d2l import torch as d2l
d2l.use_svg_display()

def Nin_block(in_channel,out_channel,stride,padding,kernal_size):
    block_list=[]
    block_list.append(nn.Conv2d(in_channel,out_channel,stride=stride,padding=padding,kernel_size=kernal_size))
    block_list.append(nn.ReLU())
    block_list.append(nn.Conv2d(out_channel,out_channel,kernel_size=1))
    block_list.append(nn.ReLU())
    block_list.append(nn.Conv2d(out_channel,out_channel,kernel_size=1))
    block_list.append(nn.ReLU())
    return nn.Sequential(*block_list)

def Nin_net():
    Nin_list=[]
    Nin_list.append(Nin_block(in_channel=1,out_channel=96,stride=4,padding=0,kernal_size=11))
    Nin_list.append(nn.MaxPool2d(kernel_size=3,stride=2))
    Nin_list.append(Nin_block(in_channel=96,out_channel=256,stride=1,padding=2,kernal_size=5))
    Nin_list.append(nn.MaxPool2d(kernel_size=3, stride=2))
    Nin_list.append(Nin_block(in_channel=256,out_channel=384,stride=1,padding=1,kernal_size=3))
    Nin_list.append(nn.MaxPool2d(kernel_size=3, stride=2))
    Nin_list.append(nn.Dropout(0.5))  #nn.Dropout(0)的时候test的准确率=0.86保持正常，但是train的训练速度（最后为0.71）放慢了，（）侧面证明了drop_out只做用于训练集
    Nin_list.append(Nin_block(in_channel=384,out_channel=10,stride=1,padding=1,kernal_size=3))
    Nin_list.append(nn.AdaptiveAvgPool2d((1,1)))   #会使得迭代的速度变慢，因此相对于Alex net 会增加一些学习率
    Nin_list.append(nn.Flatten())
    return nn.Sequential(*Nin_list)

net=Nin_net()

'''
X=torch.randn((1,1,224,224),dtype=torch.float32)
for layer in net:
    X=layer(X)
    print(layer.__class__.__name__,'\toutput shape:',X.shape)
'''

class add_machine():
    def __init__(self,n):
        self.data=[0.0]*n
    def add(self,*args):
        self.data=[i+float(j) for i,j in zip(self.data,args)]
    def __getitem__(self, item):
        return self.data[item]

def accuracy(y_hat,y):
    y_hat=torch.argmax(y_hat,dim=1)
    cmp=(y_hat.type(y.dtype)==y)
    return cmp.type(y.dtype).sum()

def evaluate_accuracy(net,deal_data,device):
    if isinstance(net,nn.Module):
        net.eval()
    matric=add_machine(2)
    for X,y in deal_data:
        if isinstance(X,list):
            X=[x.to(device) for x in X]
        else:
            X=X.to(device)
        y=y.to(device)
        y_hat=net(X)
        matric.add(accuracy(y_hat,y),y.numel())
    return matric[0]/matric[1]

def train(net,train_iters,test_iters,epoch_num,lr,device):
    def init_params(m):
        if type(m)==nn.Linear or type(m)==nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_params)
    loss=nn.CrossEntropyLoss()
    trainer=torch.optim.SGD(net.parameters(),lr=lr)
    net.to(device)
    print('is training : ',device)
    matric=add_machine(3)
    for epoch in range(epoch_num):
        net.train()
        for X,y in train_iters:
            X,y=X.to(device),y.to(device)
            y_hat=net(X)
            l=loss(y_hat,y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
            matric.add(l*y.shape[0],accuracy(y_hat,y),y.numel())
        test_acc=evaluate_accuracy(net,test_iters,device)
        print('当前迭代次数为:',epoch+1)
        print('train loss:',matric[0]/matric[2])
        print('train acc:', matric[1] / matric[2])
        print('test acc:', test_acc)
        print('*'*50)


def try_gpu():
    if torch.cuda.device_count()>=1:
        return torch.device('cuda')
    else:
        return torch.device('cpu')

batch_size=128
train_iters,test_iters=d2l.load_data_fashion_mnist(batch_size,resize=224)
train(net,train_iters,test_iters,epoch_num=10,lr=0.1,device=try_gpu())