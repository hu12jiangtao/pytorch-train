from torch import nn
import torch
import torchvision
import util
from torchvision import datasets
from torch.utils import data
import Models
import os
import numpy as np
import random

# 在训练过程中过程,此时的queue=[128, 65535]
# 此时在数据集中随机选择一个批量(得到了query=[batch,3,32,32],positive_key=[batch,3,32,32])
# query经过encode_q得到对应向量f_q=[batch,128],key经过了encode_k得到了对应向量f_k=[batch,128]
# 计算f_q和 queue负样本以及f_k正样本之间的相似度(点积),
# 负样本计算: dis_neg = mm(f_q,queue)=[batch,65536] 正样本计算dis_pos = bmm(f_q.unsqueeze(1),f_q.unsqueeze(2))=[batch,1]
# 此时在y_hat = concat(dis_pos,dis_neg)
# 此时将y_hat除以温度系数后进行交叉熵损失计算后更新encode_q，再根据encode_q更新encode_k
# 之后将f_k加入到queue中，在删除queue最开始的一个batch的特征

class Config(object):
    def __init__(self):
        self.device = torch.device('cuda:0')
        self.batch_size = 256
        self.MODEL = 'wide_resnet'
        self.ALL_EPOCH = 180
        self.save_model_path = 'checkpoint_out/wide_resnet'
        self.QUEUE_LENGTH = 4096
        self.t = 0.07

def get_model(model_name, config): # 用来获取网络的结构的
    try:
        if model_name == 'wide_resnet':
            encode_q = Models.WideResNet(pretrained=None)
            encode_k = Models.WideResNet(pretrained=None)
            # 损失函数的更新只更新encode_q，不更新encode_k
            for param in encode_k.parameters():
                param.requires_grad = False
            # 为了实现再利用MoCo进行预训练时选用的初始梯度参数和有监督的预训练的初始参数相近似，需要shuffle BN
            device_lst = [config.device] * 4
            # 使用多个GPU对模型进行训练的过程(以三个GPU举例):
            # 1.首先将整个训练模型转移到GPU0上，之后将模型复制到GPU1、GPU2上面
            # 2.将输入模型的数据转移到GPU0上，并且按照顺序将数据均分到每个GPU的上面
            # 3.在每个GPU上的到输出的结果，之后将输出的结果在GPU0上concat起来，之后进行损失的计算
            encode_q = nn.DataParallel(encode_q,device_ids=device_lst)
            encode_k = nn.DataParallel(encode_k,device_ids=device_lst) # 此时的model的参数仍然存在在cpu上面
            encode_q = encode_q.to(config.device) # 再利用多卡进行训练的时候需要首先将数据和模型转移到第一张GPU上，之后进行数据分配
            encode_k = encode_k.to(config.device)
            return encode_q,encode_k
    except KeyError:
        print(f'model name:{model_name} does not exist.')

def momentum_update(encode_q, encode_k, momentum):
    for param_q,param_k in zip(encode_q.parameters(),encode_k.parameters()):
        param_k.data.mul_(momentum).add_(1 - momentum, param_q.detach().data)

def get_shuffle_idx(batch_size, device):
    # 此时的reverse_index[0]代表着原先的第0个样本在shuffle_index上的索引
    shuffle_index = [i for i in range(batch_size)]
    random.shuffle(shuffle_index)
    shuffle_index = np.array(shuffle_index)
    reverse_index = np.argsort(shuffle_index)
    return torch.as_tensor(shuffle_index,dtype=torch.long,device=device),\
           torch.as_tensor(reverse_index,dtype=torch.long,device=device)

def enqueue(queue, k): # 加入新的batch的特征
    # queue=[4096,128], k=[256,128]
    return torch.cat([queue,k],dim=0)

def dequeue(queue,config):
    if queue.shape[0] < config.QUEUE_LENGTH:
        return queue
    else:
        return queue[-config.QUEUE_LENGTH:]

class AddMachine:
    def __init__(self,n):
        self.data = [0.] * n
    def add(self,*args):
        self.data = [i + float(j) for i,j in zip(self.data,args)]
    def __getitem__(self, item):
        return self.data[item]

def accuracy(y_hat, y):
    a = torch.argmax(y_hat,dim=-1)
    cmp = (a.type(y.dtype) == y)
    return cmp.type(y.dtype).sum()

def train(train_loader, model_q, model_k, queue, opt, criterion, config):
    # queue.shape=[4096,128]，queue的长度应该是可以被batch_size整除的
    model_q.train()
    model_k.train()
    metric = AddMachine(4)
    for img_q, img_k, _ in train_loader:
        if queue is not None and queue.shape[0] == config.QUEUE_LENGTH:
            # 此时将logits中的/config.t拿去同正确的logits中的/config.t拿去的值相同(差别在0.0001大小)
            # 这里出问题了(logits计算出现了错误)
            img_q, img_k = img_q.to(config.device), img_k.to(config.device)
            q = model_q(img_q) # q = [batch, 128]
            # 进行shuffle BN操作(类似于gather的操作), 此时的目标为打乱批量的顺序，计算完特征后又恢复原来的输入图片的顺序
            shuffle_index,reverse_index = get_shuffle_idx(config.batch_size, config.device)
            img_k = img_k[shuffle_index]
            k = model_k(img_k)
            k = k[reverse_index].detach() # [batch, 128]
            # 开始进行model_q的更新
            # 1.计算q和所有k之间的点积距离
            l_pos = torch.bmm(q.unsqueeze(1),k.unsqueeze(2)).reshape(q.shape[0],-1) # [batch,1]
            l_neg = torch.matmul(q, queue.permute(1,0)) # [batch,4096]
            logits = torch.cat([l_pos,l_neg],dim=1) / config.t # [128,4097]
            label = torch.zeros(size=(logits.shape[0],),dtype=torch.long,device=config.device)
            l = criterion(logits, label)
            opt.zero_grad()
            l.backward()
            opt.step()
            metric.add(l * label.numel(), accuracy(logits,label), logits[:,0].sum(), label.numel())
            momentum_update(model_q, model_k, 0.999) # 按照论文来momentum=0.999(用来更新encode_k)
        else:
            # 当queue中的元素没有达到标准的时候，不会对encode_q进行训练(此时也需要shuffle BN)
            img_k = img_k.to(config.device)
            shuffle_index, reverse_index = get_shuffle_idx(config.batch_size, config.device)
            img_k = img_k[shuffle_index]
            k = model_k(img_k)
            k = k[reverse_index] # [256,128]  # 正确的其中是由detach的，不正确的其中是没有detach的
        # 之后需要对queue进行更新
        queue = enqueue(queue, k) if queue is not None else k
        queue = dequeue(queue, config)
    return metric,queue

if __name__ == '__main__':
    config = Config()
    # 固定随机种子用来检验模型
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)
    # 导入数据集(此时创建的dataset中每一个样本应当包含经过不同的transform变换的图片1矩阵和图片2矩阵)
    # 每个样本的transform1和transform2应当尽可能的不一样
    image_size, mean, std  = util.dataset_info(name='cifar') # 获得图片的尺寸和图片的方差以及均值
    # 创建图片的数据增强方式
    train_transform = util.get_transform(image_size, mean, std, mode='train', to_tensor=True)
    # 对于每个批量的数据进行数据增强
    train_dataset = util.custom_dataset(datasets.cifar.CIFAR10)(root='D:\\python\\pytorch作业\\all_data\\cifar10\\data',
                                                                train=True, transform=train_transform,download=False)
    # train_loader经过检验得到和正确的相同
    train_loader = data.DataLoader(train_dataset,batch_size=config.batch_size,shuffle=True,drop_last=True)
    # 导入自己设计的encode模型，此时的encode_q和encode_k的模型结构相同，初始参数也相同
    model_q, model_k = get_model(config.MODEL,config)
    # 迭代过程中的参数
    opt = torch.optim.SGD(model_q.parameters(),lr=0.02, momentum=0.9, nesterov=True, weight_decay=1e-5)
    per = config.ALL_EPOCH // 6
    # 此时的学习率衰减器为在60，120，150轮迭代的时候学习率降低10倍
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[per * 2, per * 4, per * 5], gamma=0.1)
    # 损失韩式是利用交叉熵函数
    criterion = nn.CrossEntropyLoss()
    # encode_q和encode_k的初始参数(初始的时候两个encode的参数应当是相同的)
    momentum_update(model_q, model_k, 0)
    # 开始进行训练
    torch.backends.cudnn.benchmark = True
    queue = None
    start_epoch = 0
    min_loss = float('inf')
    # 开始导入已经训练好的模型
    if os.path.exists(config.save_model_path):
        model_q.load_state_dict(torch.load(config.save_model_path)['model_q'])
        model_k.load_state_dict(torch.load(config.save_model_path)['model_k'])
        opt.load_state_dict(torch.load(config.save_model_path)['opt'])
        scheduler.load_state_dict(torch.load(config.save_model_path)['scheduler'])
        start_epoch = torch.load(config.save_model_path)['epoch']
        min_loss = torch.load(config.save_model_path)['loss']
        print(f'loaded model from {config.save_model_path}')
    # # 开始进行训练(最后经过180轮的迭代后的准确率为92%)
    for epoch in range(start_epoch, config.ALL_EPOCH):
        metric, queue = train(train_loader, model_q, model_k, queue, opt, criterion, config)
        train_loss = metric[0] / metric[3]
        train_acc = metric[1] / metric[3]
        dot_positive = metric[2] / metric[3]
        print(f'epoch:{epoch + 1}, train_loss:{train_loss:1.4f}, '
              f'train_acc:{train_acc:1.4f} dot_positive:{dot_positive:1.4f}')
        scheduler.step()
        if train_loss < min_loss:
            min_loss = train_loss
            state = {'model_q':model_q.state_dict(),'model_k':model_k.state_dict(),'opt':opt.state_dict(),
                     'scheduler':scheduler.state_dict(),'epoch':epoch + 1,'loss':min_loss}
            print(f'save to {config.save_model_path}')
            torch.save(state, config.save_model_path)













