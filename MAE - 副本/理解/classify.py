# 此时的预训练已经完成了，在判别任务上利用MAE训练好的encode的参数(作为迁移学习)进行微调得到模型参数
# 此时的目标为 利用较小的数据集进行MAE预训练后在这个数据集上得到的判别准确率 比 直接利用ViT在这个数据集上直接训练得到的判别准确率要高
from torch import nn
import torch
import model
from model import ViT_Classifier
import torchvision
from torch.utils import data
import math

class Config(object):
    def __init__(self):
        self.device = torch.device('cuda')
        # mae模型参数
        self.image_size = 32
        self.patch_size = 2
        self.emb_dim = 192
        self.encode_num_layer = 12
        self.encode_num_head = 3
        self.decode_num_layer = 4
        self.decode_num_head = 3
        self.mask_ratio = 0.75
        # 迭代参数
        self.use_pretrain = False
        self.batch_size = 64
        self.num_classes = 10
        self.num_epoch = 100
        self.warmup_epoch = 5
        self.base_learning_rate = 1e-3
        self.weight_decay = 0.05

class AddMachine:
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

def evaluate_accuracy(data_loader,net,device):
    net.to(device)
    net.eval()
    metric = AddMachine(2)
    for x,y in data_loader:
        x,y = x.to(device),y.to(device)
        y_hat = net(x) # [batch,10]
        metric.add(evaluate(y_hat,y),y.numel())
    return metric[0] / metric[1]

def train(train_loader,test_loader,net,config, save_ViT_path):
    loss = nn.CrossEntropyLoss()
    # opt = torch.optim.Adam(net.parameters(),lr=3e-4)
    opt = torch.optim.AdamW(net.parameters(), lr=config.base_learning_rate * config.batch_size / 256,
                              betas=(0.9, 0.999), weight_decay=config.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (config.warmup_epoch + 1e-8),
                                0.5 * (math.cos(epoch / config.num_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_func, verbose=True)

    for i in range(config.num_epoch):
        net.train()
        metric = AddMachine(3)
        for x,y in train_loader:
            x, y = x.to(config.device), y.to(config.device)
            y_hat = net(x)
            l = loss(y_hat, y)
            opt.zero_grad()
            l.backward()
            opt.step()
            metric.add(l * y.numel(), evaluate(y_hat,y), y.numel())
        lr_scheduler.step()
        net.eval()
        test_acc = evaluate_accuracy(test_loader,net,config.device)
        print(f'epoch:{i + 1} train_loss:{metric[0] / metric[2]:1.4f} '
              f'train_acc:{metric[1] / metric[2]:1.4f} test_acc:{test_acc:1.4f}')
        torch.save(net.state_dict,save_ViT_path)

if __name__ == '__main__':
    config = Config()
    # 导入训练数据和验证数据
    trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(0.5,0.5)])
    train_dataset = torchvision.datasets.CIFAR10('data',train=True,transform=trans,download=False) # 使用的是CIFAR10数据集
    test_dataset = torchvision.datasets.CIFAR10('data',train=False,transform=trans,download=False)
    train_loader = data.DataLoader(train_dataset,config.batch_size,shuffle=True)
    test_loader = data.DataLoader(test_dataset,config.batch_size,shuffle=True)
    # 导入MAE模型
    pretrain_model = model.MAE_ViT(config.image_size,config.patch_size,config.emb_dim,config.encode_num_layer,
                          config.encode_num_head,config.decode_num_layer,config.decode_num_head,config.mask_ratio)
    pretrain_model.to(config.device)
    if config.use_pretrain is True:
        pretrain_model.load_state_dict(torch.load('param.params'))
        save_ViT_path = 'classify_pretrain.params'
    else:
        save_ViT_path = 'classify_no_pretrain.params'
    # 从MAE模型中导出ViT模型
    net = ViT_Classifier(pretrain_model.encode,config.num_classes)
    net.to(config.device)
    # 开始进行训练和验证
    train(train_loader, test_loader, net, config, save_ViT_path)
    # 只使用Adam，lr=0.001的情况下利用MAE的encode进行预训练的模型的最优的验证准确率为0.8776，此时的训练准确率为0.9709，此时的迭代次数为15
    # 只使用Adam，lr=0.001的情况下如果没有进行pretrain的时候，模型的准确率只在0.46周围徘徊
    # 在使用带有正则化Adam，以及梯度衰减的情况下 训练集的损失在不断的下降且训练数据集的准确率在不断的增加，但是测试集的准确率在之后就是在0.61徘徊
    # 在第三条的条件下当利用库中调用的transformer的block代替自己写的transformer的block的情况下，准确率可以到大github上的0.715左右



