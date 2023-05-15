# 此时的目标:用来进行预训练(预训练的目标为生成的图片越接近与原始图片越好)
from torch import nn
import torch
import torchvision
import model
from torch.utils import data
import math

class Config(object):
    def __init__(self):
        self.device = torch.device('cuda')
        # mae模型参数
        self.batch_size = 64
        self.image_size = 32
        self.patch_size = 2
        self.emb_dim = 192
        self.encode_num_layer = 12
        self.encode_num_head = 3
        self.decode_num_layer = 4
        self.decode_num_head = 3
        self.mask_ratio = 0.75
        # mae训练参数
        self.lr = 3e-4
        self.weight_decay = 0.05
        self.total_epoch = 2000 * 4
        self.warmup_epoch = 200 * 4

def train(train_loader, test_dataset, model,config): # 此时的训练的目标为 原本图片被遮掩的patch以及生成的图片的遮掩的patch之间的绝对值差值
    opt = torch.optim.AdamW(model.parameters(),lr=config.lr,betas=(0.9,0.95),weight_decay=0.05) # 等价于在Adam上加入了正则化
    # lr_func = lambda epoch: min((epoch + 1) / (config.warmup_epoch + 1e-8),
    #                             0.5 * (math.cos(epoch / config.total_epoch * math.pi) + 1))
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt,lr_lambda=lr_func, verbose=True)
    for num in range(config.total_epoch):
        model.train()
        losses = []
        for x,y in train_loader:
            x = x.to(config.device)
            predict_images,mask = model(x)
            loss = torch.mean((predict_images * mask - x * mask)**2) / (1 - config.mask_ratio)
            losses.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
        # lr_scheduler.step()
        average_loss = sum(losses) / len(losses)
        print(f'epoch:{num + 1} loss:{average_loss:1.4f}')
        print('lr:',opt.param_groups[0]['lr'])
        # 对于每次迭代生成的图片进行检测(观测测试集中前16张生成图片)
        if (num + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_img = torch.stack([test_dataset[i][0] for i in range(16)],dim=0) # [16,3,28,28]
                val_img = val_img.to(config.device)
                eval_predict_images,eval_mask = model(val_img)
                # 此时对于未被遮掩的部分用真实图片的patches，遮掩的部分用生成的patches
                eval_predict_images = eval_predict_images * eval_mask + val_img * (1 - eval_mask)
                # 一共三张图片第一张为输入encode的所有patches，第二张是生成的图片，第三张是真实的图片
                imgs = torch.cat([val_img * (1 - eval_mask), eval_predict_images, val_img],dim=0)
                torchvision.utils.save_image(imgs,f'images/{num + 1}.png',nrow=16,normalize=True)
        torch.save(model.state_dict(),'param.params')

if __name__ == '__main__':
    config = Config()
    # 创建数据集
    trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(0.5,0.5)])
    # train_dataset = torchvision.datasets.CIFAR10('D:\\python\\pytorch作业\\all_data\\cifar10\\data',
    #                                              train=True,transform=trans,download=False)
    # test_dataset = torchvision.datasets.CIFAR10('D:\\python\\pytorch作业\\all_data\\cifar10\\data',
    #                                             train=False,transform=trans,download=False)
    train_dataset = torchvision.datasets.CIFAR10('data',train=True,transform=trans,download=False)
    test_dataset = torchvision.datasets.CIFAR10('data',train=False,transform=trans,download=False)
    # 在训练的过程中(自训练模型)，因此没用到训练数据集的标签
    train_loader = data.DataLoader(train_dataset,config.batch_size,shuffle=True)
    # 导入MAE模型
    model = model.MAE_ViT(config.image_size,config.patch_size,config.emb_dim,config.encode_num_layer,
                          config.encode_num_head,config.decode_num_layer,config.decode_num_head,config.mask_ratio)
    model.to(config.device)
    # 开始进行训练
    train(train_loader, test_dataset, model, config)

