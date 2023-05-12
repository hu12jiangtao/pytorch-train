# 对于一般的神经网络来说，并不是模型的深度越深，模型的精度越大，但是残差网络可以使你的深度变深的同时你的精度不会因此下降
# 对于残差网络来说同样最后使用的是全局平均池化层
# 残差网络的核心理念就是引入了一个正向反馈，残差块的输出g(x)=f(x)+x,x为残差块的输入，
# 当整个模型发现训练f(x)对于模型的训练没有什么影响，那么此时损失函数做梯度反传的时候梯度的变换就会很小，几乎为0，相当于这个就是一个死的模块（不参与更新）
# 在残差网络中训练精度可能比测试精度低
# 一个小的残差模块可以分为两类，第一种是输入通道等于输出通道，第二种是输入通道不等于输出通道（此时需要利用1*1的卷积块来改变输入的通道数）
# 在一个残差快中第一个卷积快一般用来修改figure的宽高，或者是通道数，第二个卷积层一般是figure_size保持不变,通道数不变

# 残差网络中最小的模块是残差块，残差块中可以分为通道数改变的和通道数不改变的，在通道数改变的时候一般是残差块中第一个卷积造成
# 残差网络中的stage中包含着最开始使图形figure减半的残差块（中的第一个卷积作用）以及后面n个不改变figure_size的残差块（除了第一个stage，原因是前面的池化层已经减半了）
# 且在残差网络的修改，我们只修改stage的深度，其他的一般不会修改

import torch
from torch import nn
from torch.nn import functional as F
import torch
from d2l import torch as d2l

#在构建网络的时候nn.Sequential中可以继续嵌套nn.Sequential当作一个子模块
d2l.use_svg_display()
class Resitual(nn.Module):  #定义一个残差块（此时当作一个层来进行定义）
    def __init__(self,in_channal,out_channal,stride=1,use_1x1conv=False):
        super().__init__()
        self.conv1=nn.Conv2d(in_channal,out_channal,kernel_size=3,padding=1,stride=stride)
        self.conv2=nn.Conv2d(out_channal,out_channal,kernel_size=3,padding=1)
        if use_1x1conv:
            self.conv3=nn.Conv2d(in_channal,out_channal,padding=1,kernel_size=3,stride=stride)
        else:
            self.conv3=None
        self.batch_norm1=nn.BatchNorm2d(out_channal) #这边设置两次batch_norm的原因是有两个层，就是代表有两个nn.BatchNorm2d类的实例
        self.batch_norm2=nn.BatchNorm2d(out_channal)

    def forward(self,X):
        Y=F.relu(self.batch_norm1(self.conv1(X)))
        Y=self.batch_norm2(self.conv2(Y))
        if self.conv3:
            X=self.conv3(X)
        Y=Y+X
        return F.relu(Y)

def resnet_stage(in_channal,out_channal,block_num,first_block=False):  #定义一个stage（每个stage中figture_Size减半，通道数加深一倍）
    blk=[]
    for i in range(block_num):
        if i==0 and first_block==False:
            blk.append(Resitual(in_channal,out_channal,stride=2,use_1x1conv=True))
        else:
            blk.append(Resitual(out_channal,out_channal,stride=1,use_1x1conv=False))
    return blk


b1=nn.Sequential(nn.Conv2d(1,64,kernel_size=7,padding=3,stride=2),
                 nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
b2=nn.Sequential(*resnet_stage(in_channal=64,out_channal=64,block_num=2,first_block=True))
b3=nn.Sequential(*resnet_stage(in_channal=64,out_channal=128,block_num=2))
b4=nn.Sequential(*resnet_stage(in_channal=128,out_channal=256,block_num=2))
b5=nn.Sequential(*resnet_stage(in_channal=256,out_channal=512,block_num=2))
net=nn.Sequential(b1,b2,b3,b4,b5,nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),nn.Linear(512,10))  #这个是ResNet-18(18个卷积层加先行层)

'''
X=torch.randn((1,1,224,224))
for layer in net:
    X=layer(X)
    print(layer.__class__.__name__,'\toutput shape:',X.shape)
'''


batch_size=128
train_iters,test_iters=d2l.load_data_fashion_mnist(batch_size,resize=96)

def try_gpu():
    if torch.cuda.device_count()>=1:
        return torch.device('cuda')
    else:
        return torch.device('cpu')

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
    print('is training:',device)
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

        test_acc=evaluate_accuracy(net, test_iters, device)
        print('当前迭代次数:',epoch+1)
        print('train loss:',matric[0]/matric[2])
        print('train acc:', matric[1] / matric[2])
        print('test acc:', test_acc)
        print('*'*50)


train(net, train_iters, test_iters, epoch_num=10, lr=0.05, device=try_gpu())
