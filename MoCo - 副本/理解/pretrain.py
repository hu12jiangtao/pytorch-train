# 此时利用MoCo训练好的模型将图片转化为特征之后微调获得判别准确率(冻结骨干网络)
from torch import nn
import torch
import torchvision
from torch.utils import data
import main

class Config(object):
    def __init__(self):
        self.batch_size = 256
        self.feature_num = 128
        self.class_num = 10
        self.device = torch.device('cuda:0')
        self.num_epoch = 100
        self.pretrain_mode = 'ResNet'

class MoCo_Net(nn.Module):
    def __init__(self,feature_num,class_num):
        super(MoCo_Net, self).__init__()
        config1 = main.Config()
        self.model_q,_ = main.get_model('wide_resnet',config1)
        self.model_q.load_state_dict(torch.load('checkpoint_out/wide_resnet')['model_q']) # 导入模型
        self.fc = nn.Linear(feature_num,class_num)

    def forward(self,x):
        out = self.model_q(x) # [batch,128]
        out = out.detach()
        return self.fc(out)

class Res_Net(nn.Module):
    def __init__(self,class_num):
        super(Res_Net, self).__init__()
        pretrain_model = torchvision.models.resnet18(pretrained=True)
        self.bone = nn.Sequential(*list(pretrain_model.children())[:-1])
        self.fc = nn.Linear(pretrain_model.fc.in_features, class_num)
    def forward(self,x):
        out = self.bone(x)
        out = out.reshape(out.shape[0],-1)
        out = out.detach()
        return self.fc(out)

class AddMachine:
    def __init__(self,n):
        self.data = [0.] * n
    def add(self,*args):
        self.data = [i + float(j) for i,j in zip(self.data,args)]
    def __getitem__(self, item):
        return self.data[item]

def accuracy(y_hat,y):
    a = torch.argmax(y_hat,dim=-1)
    cmp = (a.type(y.dtype) == y)
    return cmp.type(y.dtype).sum()

def evaluate_accuracy(data_loader,net,device):
    net.to(device)
    net.eval()
    metric = AddMachine(2)
    for x,y in data_loader:
        x,y = x.to(device), y.to(device)
        y_hat = net(x)
        metric.add(accuracy(y_hat,y),y.numel())
    return metric[0] / metric[1]

def train(train_loader,test_loader,net,config,mode):
    opt = torch.optim.Adam([{'params':net.fc.parameters(),'lr':3e-4}])
    loss = nn.CrossEntropyLoss()
    for epoch in range(config.num_epoch):
        net.train()
        metric = AddMachine(3)
        for x,y in train_loader:
            x, y = x.to(config.device), y.to(config.device)
            y_hat = net(x)
            l = loss(y_hat,y)
            opt.zero_grad()
            l.backward()
            opt.step()
            metric.add(l * y.numel(), accuracy(y_hat, y), y.numel())
        test_acc = evaluate_accuracy(test_loader,net,config.device)
        print(f'epoch:{epoch + 1} train_loss:{metric[0]/metric[2]:1.3f} '
              f'train_acc:{metric[1]/metric[2]:1.3f} test_acc:{test_acc:1.3f}')
        torch.save(net.state_dict(),f'checkpoint_out/{mode}.params')



if __name__ == '__main__':
    config = Config()
    # 准备数据集
    test_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Normalize
                                                     (mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))])
    train_transform = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
                                                      torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize
                                                      (mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))])

    train_dataset = torchvision.datasets.CIFAR10(root='D:\\python\\pytorch作业\\all_data\\cifar10\\data',
                                                 train=True,transform=train_transform,download=False)
    test_dataset = torchvision.datasets.CIFAR10(root='D:\\python\\pytorch作业\\all_data\\cifar10\\data',
                                                 train=False,transform=test_transform,download=False)
    train_iter = data.DataLoader(train_dataset,batch_size=config.batch_size,shuffle=True)
    test_iter = data.DataLoader(test_dataset,batch_size=config.batch_size,shuffle=False)

    # 开始进行微调
    print(f'start training {config.pretrain_mode} model')
    if config.pretrain_mode == 'MoCo':
        # 判别模型(冻结骨干网络)
        net = MoCo_Net(config.feature_num, config.class_num).to(config.device)
        # 利用MoCo模型作为预训练模型的微调(冻结骨干网络)的准确率在0.59左右(train_acc:0.585 test_acc:0.59)
    elif config.pretrain_mode == 'ResNet':
        net = Res_Net(config.class_num).to(config.device)
        # 利用ResNet模型作为预训练模型的微调(冻结骨干网络)的准确率在0.59左右(train_acc:0.486 test_acc:0.475)

    train(train_iter, test_iter, net, config, mode=config.pretrain_mode)

