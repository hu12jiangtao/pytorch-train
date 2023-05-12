import os

import torch
from torch import nn
import torchvision
from torch.utils import data

class Config(object):
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_batch_size = 64
        self.test_batch_size = 4
        self.data_dir = 'dataset'
        # 'D:\\python\\pytorch作业\\知识蒸馏\\my_self_cifir10\\dataset'为自己电脑上,'dataset'为云上训练的
        self.momentum = 0.9
        self.lr = 0.001
        self.num_epoch = 10
        self.model_save_dir = f'param/resnet18.pkl'


def load_cifir_data(path,train_batch_size,test_batch_size):
    # RandomCrop中的padding代表的含义为首先将整个图片进行padding之后在进行随机的裁剪
    train_trans = torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)),
                                                  torchvision.transforms.RandomHorizontalFlip(0.5),
                                                  torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                                                                   [0.2023, 0.1994, 0.2010])])
    test_trans = torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)),
                                                 torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                                                                  [0.2023, 0.1994, 0.2010])])
    train_data = torchvision.datasets.CIFAR10(path,train=True,download=False,transform=train_trans)
    test_data = torchvision.datasets.CIFAR10(path,train=False,download=False,transform=test_trans)
    train_loader = data.DataLoader(train_data,batch_size=train_batch_size,shuffle=True)
    test_loader = data.DataLoader(test_data,batch_size=test_batch_size,shuffle=False)
    return train_loader,test_loader

class add_machine(object):
    def __init__(self,n):
        self.data = [0.] * n
    def add(self,*args):
        self.data = [i + float(j) for i,j in zip(self.data,args)]
    def __getitem__(self, item):
        return self.data[item]

def accuracy(y,y_hat):
    a = torch.argmax(y_hat,dim=-1)
    cmp = (a.type(y.dtype) == y)
    return cmp.type(y.dtype).sum()

def evaluate(net,data_loader,device):
    metric = add_machine(2)
    net.eval()
    net.to(device)
    for X,y in data_loader:
        X,y = X.to(device),y.to(device)
        y_hat = net(X)
        metric.add(accuracy(y,y_hat),y.numel())
    return metric[0] / metric[1]

def train_epoch(epoch,net,train_loader,test_loader,loss,opt,config):
    net.train()
    metric = add_machine(3)
    for index,(X,y) in enumerate(train_loader):
        X,y = X.to(config.device),y.to(config.device)
        y_hat = net(X)
        l = loss(y_hat,y)
        opt.zero_grad()
        l.backward()
        opt.step()
        metric.add(l * y.numel(), accuracy(y,y_hat), y.numel())
    test_acc = evaluate(net,test_loader,config.device)
    print(f'[epoch:{epoch + 1} \t train_loss:{metric[0] / metric[2]:1.3f} \t train_acc:{metric[1] / metric[2]:1.3f} \t test_acc:{test_acc}')



if __name__ == '__main__':
    # 创建数据集
    torch.manual_seed(1)
    config = Config()
    torch.backends.cudnn.benchmark = True
    # 导入数据集(train:[256,3,32,32],test:[100,3,32,32])
    train_loader, test_loader = load_cifir_data(config.data_dir,config.train_batch_size,config.test_batch_size)
    # 网络
    net = torchvision.models.resnet18(pretrained=True)
    net.fc = nn.Linear(net.fc.in_features, 10)
    net.to(config.device)
    # 训练
    if not os.path.exists(config.model_save_dir):
        print('start training')
        loss = nn.CrossEntropyLoss()
        opt = torch.optim.SGD(net.parameters(), lr=config.lr, momentum=config.momentum)
        for epoch in range(config.num_epoch):
            train_epoch(epoch, net, train_loader, test_loader, loss, opt, config)
        torch.save(net.state_dict(), config.model_save_dir)
    else:
        net.load_state_dict(torch.load(config.model_save_dir))

    print('test_acc:',evaluate(net,test_loader,config.device)) # 0.9548
    print('train_acc:', evaluate(net, train_loader, config.device)) # 0.99932
