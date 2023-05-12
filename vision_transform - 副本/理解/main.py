from einops.layers.torch import Rearrange
import torch
import torchvision
from torchvision import transforms
from torch.nn import functional as F
from model import ViT_small
from model import ViT
from torch import nn

class Config(object):
    def __init__(self):
        self.net_name = 'tiny_vit'
        self.batch_size = 512
        self.size = 32
        self.patch_size = 4
        self.num_classes = 10
        self.dim = 512
        self.dropout = 0.1
        self.emb_dropout = 0.1
        self.lr = 1e-4
        self.n_epochs = 500
        self.device = torch.device('cuda')

class Add_machine:
    def __init__(self,n):
        self.data = [0.] * n
    def add(self,*args):
        self.data = [i + float(j) for i,j in zip(self.data,args)]
    def __getitem__(self, item):
        return self.data[item]

def evaluate(y_hat,y):
    a = torch.argmax(y_hat,dim=-1)
    cmp = (a.type(y.dtype) == y)
    return cmp.type(y.dtype).sum()

def evaluate_accuracy(data_loader, net, device):
    net.eval()
    net.to(device)
    metric = Add_machine(2)
    for x,y in data_loader:
        x, y = x.to(device), y.to(device)
        y_hat = net(x)
        metric.add(evaluate(y_hat, y),y.numel())
    return metric[0] /metric[1]


def train(train_loader, test_loader, net,loss, opt, scheduler, config):
    for i in range(config.n_epochs):
        net.train()
        metric = Add_machine(3)
        for x,y in train_loader:
            x, y = x.to(config.device), y.to(config.device)
            y_hat = net(x)
            l = loss(y_hat,y)
            opt.zero_grad()
            l.backward()
            opt.step()
            metric.add(l * y.numel(), evaluate(y_hat,y),y.numel())
        eval_acc = evaluate_accuracy(test_loader,net,config.device)
        print(f'epoch:{i+1} train_loss:{metric[0] / metric[2]:1.3f} train_acc:{metric[1] / metric[2]} test_acc:{eval_acc:1.3f}')
        scheduler.step()
        torch.save(net.state_dict(),'param.params')

if __name__ == '__main__':
    config = Config()
    # 导入数据集
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(config.size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(config.size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # train_set = torchvision.datasets.CIFAR10(root='D:\\python\pytorch作业\\all_data\\cifar10/data',
    #                                         train=True, download=False, transform=transform_train) # CIFAR10的图片大小为3*32*32
    # test_set = torchvision.datasets.CIFAR10(root='D:\\python\pytorch作业\\all_data\\cifar10/data',
    #                                         train=False, download=False, transform=transform_test) # CIFAR10的图片大小为3*32*32
    train_set = torchvision.datasets.CIFAR10(root='data',train=True, download=False, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root='data',train=False, download=False, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=config.batch_size, shuffle=True)
    # 导入模型
    if config.net_name == 'tiny_vit': # 得到结果(经过200轮迭代得到验证集准确率77.2%)
        net = ViT(32, config.patch_size, config.num_classes, config.dim, 4, 6, 256,
                  pool = 'cls', channels = 3, dim_head = 64, dropout = config.dropout, emb_dropout=config.emb_dropout)
    elif config.net_name == 'small_vit':
        net = ViT(32, config.patch_size, config.num_classes, config.dim, 6, 8, 512,
                  pool = 'cls', channels = 3, dim_head = 64, dropout = config.dropout, emb_dropout=config.emb_dropout)
    elif config.net_name == 'vit': # 改变了输入的尺寸
        net = ViT(48, config.patch_size, config.num_classes, config.dim, 6, 8, 512,
                  pool = 'cls', channels = 3, dim_head = 64, dropout = config.dropout, emb_dropout=config.emb_dropout)

    net.to(config.device)
    # 模型迭代
    loss = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(),lr=config.lr)
    # 学习率衰减
    # 经过n_epochs迭代后原先的学习率降至0, 再经过n_epochs后学习率由0升至原先的值(按照余弦函数的π/2 - 2π/3周期的图形)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,config.n_epochs) # 余弦的迭代衰减函数
    # 进行模型的学习和存储(进行一次模型参数的迭代进行一次模型的验证)
    train(train_loader, test_loader, net, loss, opt, scheduler, config)
    # 最后得到的测试集上准确率为77.2，训练集上的准确率为81.3，出现了过拟合的现象
    # 之前讲的transformer没有出现饱和是指当模型和数据集同时扩大的时候，模型的判别准确率在不断的增大，但是当数据集小模型大的情况下模型仍然会出现过拟合






