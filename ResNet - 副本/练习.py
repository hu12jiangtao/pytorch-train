import torch
from torch import nn
from d2l import torch as d2l
d2l.use_svg_display()
'''
class renet_block(nn.Module):
    def __init__(self,in_channal,out_channal,stride,is_skip):
        super(renet_block, self).__init__()
        self.conv1=nn.Conv2d(in_channal,out_channal,kernel_size=3,padding=1,stride=stride)
        self.conv2=nn.Conv2d(out_channal,out_channal,kernel_size=3,padding=1,stride=1)
        self.bn1=nn.BatchNorm2d(out_channal)
        self.bn2=nn.BatchNorm2d(out_channal)
        self.relu1=nn.ReLU()
        self.relu2=nn.ReLU()
        self.relu3=nn.ReLU()
        if is_skip:
            self.conv3=nn.Conv2d(in_channal,out_channal,kernel_size=3,padding=1,stride=stride)
        else:
            self.conv3=None
    def forward(self,X):
        Y=self.bn1(self.conv1(X))
        Y=self.relu1(Y)
        Y=self.bn2(self.conv2(Y))
        if self.conv3 is None:
            out = Y+X
        else:
            out=self.conv3(X) + Y
        return self.relu3(out)

def resnet_stage(in_channal,out_channal,num_block,is_first=False):  # num_block代表一个stage含有多少个blk，is_first代表是否是最开始的stage
    # in_channal,out_channal代表了一个stage的输入通道和输出通道
    out = []
    for i in range(num_block):
        if i == 0 and is_first == True: # 此时宽高不变通道数不变 ，此时输入通道数等于输出通道数
            out.append(renet_block(in_channal,in_channal,stride=1,is_skip=False))
        elif i == 0 and is_first == False: # 宽高减半，通道数增加
            out.append(renet_block(in_channal, out_channal, stride=2, is_skip=True))
        else: # 此时宽高不变通道数不变
            out.append(renet_block(in_channal, out_channal, stride=1, is_skip=False))
        in_channal = out_channal
    return out

b0=nn.Sequential(nn.Conv2d(in_channels=1,out_channels=64,kernel_size=7,padding=3,stride=2)
                 ,nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(kernel_size=3,padding=1,stride=2))
b1=nn.Sequential(*resnet_stage(in_channal=64,out_channal=64,num_block=2,is_first=True))
b2=nn.Sequential(*resnet_stage(in_channal=64,out_channal=128,num_block=2,is_first=False))
b3=nn.Sequential(*resnet_stage(in_channal=128,out_channal=256,num_block=2,is_first=False))
b4=nn.Sequential(*resnet_stage(in_channal=256,out_channal=512,num_block=2,is_first=False))
b5=nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),nn.Linear(512,10))
net=nn.Sequential(b0,b1,b2,b3,b4,b5)
'''
class Resitual(nn.Module):
    def __init__(self,in_channal,out_channal,stride,use_1x1conv):
        super(Resitual, self).__init__()
        self.conv1=nn.Conv2d(in_channal,out_channal,kernel_size=3,padding=1,stride=stride)
        self.conv2=nn.Conv2d(out_channal,out_channal,kernel_size=3,padding=1,stride=1)
        self.bn1=nn.BatchNorm2d(out_channal)
        self.bn2=nn.BatchNorm2d(out_channal)
        self.relu1=nn.ReLU()
        self.relu2=nn.ReLU()
        self.relu3=nn.ReLU()
        if use_1x1conv:
            self.conv3=nn.Conv2d(in_channal,out_channal,kernel_size=3,padding=1,stride=stride)
        else:
            self.conv3=None
    def forward(self,X):
        Y=self.bn1(self.conv1(X))
        Y=self.relu1(Y)
        Y=self.bn2(self.conv2(Y))
        if self.conv3 is None:
            out = Y+X
        else:
            out=self.conv3(X) + Y
        return self.relu3(out)
def resnet_stage(in_channal,out_channal,block_num,first_block=False):  #定义一个stage（每个stage中figture_Size减半，通道数加深一倍）
    out = []
    for i in range(block_num):
        if i == 0 and first_block == True: # 此时宽高不变通道数不变 ，此时输入通道数等于输出通道数
            out.append(Resitual(in_channal,in_channal,stride=1,use_1x1conv=False))
        elif i == 0 and first_block == False: # 宽高减半，通道数增加
            out.append(Resitual(in_channal, out_channal, stride=2, use_1x1conv=True))
        else: # 此时宽高不变通道数不变
            out.append(Resitual(in_channal, out_channal, stride=1, use_1x1conv=False))
        in_channal = out_channal
    return out

b0=nn.Sequential(nn.Conv2d(in_channels=1,out_channels=64,kernel_size=7,padding=3,stride=2)
                 ,nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(kernel_size=3,padding=1,stride=2))
b1=nn.Sequential(*resnet_stage(in_channal=64,out_channal=64,block_num=2,first_block=True))
b2=nn.Sequential(*resnet_stage(in_channal=64,out_channal=128,block_num=2,first_block=False))
b3=nn.Sequential(*resnet_stage(in_channal=128,out_channal=256,block_num=2,first_block=False))
b4=nn.Sequential(*resnet_stage(in_channal=256,out_channal=512,block_num=2,first_block=False))
b5=nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),nn.Linear(512,10))
net=nn.Sequential(b0,b1,b2,b3,b4,b5)

def try_gpu():
    if torch.cuda.device_count()>=1:
        return torch.device('cuda')
    else:
        return torch.device('cpu')

batch_size=128
train_iters,test_iters=d2l.load_data_fashion_mnist(batch_size,resize=96)
device=try_gpu()

class add_machine():
    def __init__(self,n):
        self.data=[0.0]*n
    def add(self,*args):
        self.data=[i+float(j) for i,j in zip(self.data,args)]
    def __getitem__(self, item):
        return self.data[item]

def accuracy(y_hat,y):
    a=torch.argmax(y_hat,dim=1)
    cmp=(a.type(y.dtype)==y)
    return cmp.type(y.dtype).sum()

def evaluate_accuracy(net,data_iter,device):
    net.eval()
    metric=add_machine(2)
    for X,y in data_iter:
        X,y=X.to(device),y.to(device)
        y_hat=net(X)
        metric.add(accuracy(y_hat,y),y.numel())
    return metric[0]/metric[1]

'''
def train(data_iter,lr,num_epochs,device):
    net.to(device)
    net.train()
    def init_param(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_param)
    loss=nn.CrossEntropyLoss()
    trainer=torch.optim.SGD(net.parameters(),lr=lr)
    for epoch in range(num_epochs):
        metric = add_machine(3)
        for X,y in data_iter:
            X,y=X.to(device),y.to(device)
            y_hat=net(X)
            l=loss(y_hat,y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
            metric.add(l,accuracy(y_hat,y),y.numel())
        acc_valid = evaluate(net,test_iters,device)
        print('当前迭代次数:', epoch + 1)
        print('train loss:', metric[0] / metric[2])
        print('train acc:', metric[1] / metric[2])
        print('test acc:', acc_valid)
        print('*' * 50)

train(train_iters,lr=0.05,num_epochs=10,device=device)
'''
def train(net,train_iters,test_iters,epoch_num,lr,device):
    def init_params(m):
        if type(m)==nn.Linear or type(m)==nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_params)
    loss=nn.CrossEntropyLoss()
    trainer=torch.optim.SGD(net.parameters(),lr=lr)
    net.to(device)
    print('is training:',device)
    for epoch in range(epoch_num):
        net.train()  # 如果放在最前面会导致当进行了一次的迭代后网络变成了net.eval()型会导致出错（evaluate_accuracy中变换的）
        metric = add_machine(3)
        for X,y in train_iters:
            X,y=X.to(device),y.to(device)
            y_hat=net(X)
            l=loss(y_hat,y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
            metric.add(l*y.shape[0], accuracy(y_hat, y), y.numel())

        test_acc=evaluate_accuracy(net, test_iters, device)
        print('当前迭代次数:',epoch+1)
        print('train loss:',metric[0]/metric[2])
        print('train acc:', metric[1] / metric[2])
        print('test acc:', test_acc)
        print('*'*50)


train(net, train_iters, test_iters, epoch_num=10, lr=0.05, device=try_gpu())