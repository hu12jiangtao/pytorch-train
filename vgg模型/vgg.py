# 对于vgg-x模型的x，x就是全连接层的3，加上卷积层的个数
# vgg提出了深度神经网络中块的概念，是一种对Alexnet前面不规则的卷积块的一种改进，且vgg块中的卷积都是相同的kernal_size=3,stride=1和padding=1
# 最后的全连接层还是相同的，之后的神经网络可以利用多个块来构建（一个块的参数有layer_num层数，通道输入数，通道输出数），输入通道可以不等于输出通道数
# 在块中的卷积不会改变图形的像素，池化层会将宽高给减半，因此一个块会将图像的尺寸减半，对于imagenet中的224*224的尺寸来说用七个块尺寸减到7*7（vgg-5）
# 一个块包含了多个kernal_size=3,padding=1的卷积神经网络和一个kernal_size等于2stride=2 的最大池化层

import torch
from torch import nn
from d2l import torch as d2l
d2l.use_svg_display()

def try_gpu():
    if torch.cuda.device_count()>=1:
        return torch.device('cuda')
    else:
        return torch.device('cpu')

batch_size=64
train_iters,test_iters=d2l.load_data_fashion_mnist(batch_size,resize=224)

def vgg_block(layer_num,in_channal,out_channal):  # 此时in_channal规定了进入该模块后第一个卷积的输入通道数,out_channal代表着之后卷积的输入通道数和输出通道数
    vgg_list=[]
    for i in range(layer_num):
        vgg_list.append(nn.Conv2d(in_channal,out_channal,kernel_size=3,padding=1))
        vgg_list.append(nn.ReLU())
        in_channal=out_channal
    vgg_list.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*vgg_list)  #相当于将列表中的元素按顺序取出

def vgg(conv_arch):
    vgg=[]
    in_channal=1
    for layer_num,out_channal in conv_arch:
        vgg.append(vgg_block(layer_num,in_channal,out_channal))
        in_channal=out_channal  #下一个块的输入必须和上一个块的输出保持一致
    net=nn.Sequential(*vgg)
    net.add_module('flatten',nn.Flatten())
    net.add_module('Linear1',nn.Linear(7*7*out_channal,4096))
    net.add_module('activate1',nn.ReLU())
    net.add_module('drop_out1',nn.Dropout(0.5))
    net.add_module('Linear2',nn.Linear(4096,4096))
    net.add_module('activate2',nn.ReLU())
    net.add_module('drop_out2',nn.Dropout(0.5))
    net.add_module('Linear3',nn.Linear(4096,10))
    return net

conv_arch=((1,64),(1,128),(2,256),(2,512),(2,512))  # 宽高减半，通道数增加一倍  #经典的vgg-11网络
radio=8
small_conv_arch=((i,j//radio) for i,j in conv_arch)  #将所有的通道数整体除以4
net=vgg(small_conv_arch)


'''
#用来查看网络的每一层的输出
X=torch.randn((1,1,224,224),dtype=torch.float32)
for layer in net:
    X=layer(X)
    print(layer.__class__.__name__,'output shape:',X.shape)
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
    print('training in :',device)
    for epoch in range(epoch_num):
        net.train()
        matric=add_machine(3)
        for X,y in train_iters:
            X,y=X.to(device),y.to(device)
            y_hat=net(X)
            l=loss(y_hat,y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
            matric.add(l * y.shape[0], accuracy(y_hat, y), y.numel())
        test_acc = evaluate_accuracy(net, test_iters, device)
        train_loss = matric[0] / matric[2]
        train_acc = matric[1] / matric[2]
        print('当前迭代的次数:', epoch + 1)
        print('train_loss:', train_loss)
        print('train_acc:', train_acc)
        print('test_acc', test_acc)
        print('*' * 50)


train(net,train_iters,test_iters,epoch_num=10,lr=0.075,device=try_gpu())