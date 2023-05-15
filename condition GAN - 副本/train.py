# condition GAN的目标为生成一张与文字匹配的图片
# 生成器的输入:一句话 + 一个随机噪声 ; 输出: 一张生成的图片
# 判别器的输入:一句话 + 生成的图片 ; 输出:一个分数
# 生成器的目标: 将生成的图片丢入判别器中得到的分数越高越好
# 判别器的目标: 当输入的图片为假图片时输出分数越低越好，当输入的图片为真图片但是和文字不匹配时分数越低越好；当输入的图片为真同时匹配输入的文字得到的分数越高越好

# 寻常网络的架构: 一个样本中文字通过net1得到一个向量;一张图片通过net2得到一个向量，将两个向量集合在一起利用全连接得到一个分数
# 一种改进后的GAN的架构: 一张图片通过net2得到一个向量

# 为了使GAN可以进行工作:
# 1.将输入的图片进行归一化，归一化后的结果在[-1,1]之间，生成器的最后使用tanh函数
# 2.对目标函数进行修正 min(log(1-D(x))) 等价于 max(log(D(x)))
# 3.用球面线性插值代替线性插值，可以防止偏离模型的先验分布，产生更清晰的样本，详细情况可见
# 4.使用批量归一化(不能使用批量归一化就使用InstanceNorm)，对于一个batch只存在一种类型的样本，全是假样本或者全是真样本
# 5.不在使用平均池化层和relu激活层，而是利用LeakyRelu
# 6.标签平滑化:当D输出的分数在[0.7,1.2]的情况下可以认为是真样本，分数在[0,0.3]的情况下为假样本
# 7.利用全卷积的网络
# 8.利用adam函数
# 9.判别器的损失=0说明模型训练失败。如果生成器的损失稳步下降，说明判别器没有起作用
# 10.在输入端适当添加噪声：在判别器的输入中加入一些人工噪声。在生成器的每层中都加入高斯噪声。
# 11.部分时候利用1*1卷积来替代全连接层

import torch
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from torch.utils import data
import torchvision
import model
from torch import nn
from torchvision.utils import save_image


# 在TF中的tensor只是创建一个节点，但是没有将其放进计算图中
# 在pytorch中已经不需要用Variable，此时的tensor已经是Variable加入到了计算图了
# Tensor和tensor是深拷贝的，而from_numpy这个是tensor

def load_file(config):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()
                                                ])
    # 在本地进行的时候的路径为D:\\python\\gan简单版\\解决模式崩溃（模板）\\data ; 在云服务器上为data
    train_dataset = torchvision.datasets.MNIST(root='data',
        train=True,
        download=False,
        transform=transform)
    train_iter = data.DataLoader(train_dataset,batch_size=config.batch_size,shuffle=True,drop_last=True)
    return train_iter

def train(train_iter,G,D,config):  # 训练G，D的参数，之后通过噪音、以及独热话后的label生成相对应的图像
    #opt_D = torch.optim.Adam(D.parameters(),lr=1e-4,betas=(0.5, 0.999))  # Adam的学习参数为1e-3训练不了，出现了模式崩溃
    #opt_G = torch.optim.Adam(G.parameters(),lr=1e-4,betas=(0.5, 0.999)) # 对于adam的学习算法来说其所占的显存大于使用SGD所占的显存
    opt_D = torch.optim.SGD(D.parameters(), lr=1e-2)
    opt_G = torch.optim.SGD(G.parameters(), lr=1e-2)
    '''
    # 使用的是硬分类，可以使用软分类
    fake = torch.zeros(size=(config.batch_size,1),dtype=torch.float32,device=config.device)
    real = torch.ones(size=(config.batch_size,1),dtype=torch.float32,device=config.device)
    '''
    # 使用软分类
    fake = torch.tensor([np.random.uniform(0,0.2) for _ in range(config.batch_size)],device=config.device).reshape(-1,1)
    real = torch.tensor([np.random.uniform(0.9,1.1) for _ in range(config.batch_size)],device=config.device).reshape(-1,1)
    batch_iter = 0
    loss = nn.BCELoss()
    test_label = torch.repeat_interleave(torch.arange(config.NUM_LABELS,device=config.device),
                                         config.each_test_num,dim=0) # [50,]
    test_hot_label = F.one_hot(test_label,config.NUM_LABELS).type(torch.float32) # [batch,10]
    for epoch in range(config.num_epochs):
        D.train()
        G.train()
        for x,label in train_iter:
            x,label = x.to(config.device), label.to(config.device)
            # 此时首先先训练判别器，对应的标签+真实图片 得到高分 ； 不对应的标签+生成的图片 的到低分
            hot_label = F.one_hot(label,config.NUM_LABELS).type(torch.float32) # [batch, 10]
            rand_y = torch.from_numpy(np.random.randint(0, config.NUM_LABELS, size=(config.batch_size,)))
            rand_y = rand_y.to(config.device).type(torch.int64)
            fake_hot_label = F.one_hot(rand_y,config.NUM_LABELS).type(torch.float32)
            # 对于D训练
            noise = torch.randn(size=(config.batch_size,config.noise_dim),device=config.device)
            fake_gen = G(noise,fake_hot_label) # [batch,1,28,28]
            pred_gen = D(fake_gen.detach(), fake_hot_label)
            pred_real = D(x, hot_label)
            real_loss = loss(pred_real,real)
            fake_loss = loss(pred_gen,fake)
            d_loss = (real_loss + fake_loss) / 2
            opt_D.zero_grad()
            d_loss.backward(retain_graph=True)
            opt_D.step()
            # 对应的G训练
            pred_gen_G = D(fake_gen,fake_hot_label)
            g_loss = loss(pred_gen_G,real)
            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()
            batch_iter += 1
            if batch_iter % 500 == 0:
                print(f'epoch:{epoch},batch_iter:{batch_iter},d_loss:{d_loss:1.4f},g_loss:{g_loss:1.4f}')
                noise_test = torch.randn(size=(config.each_test_num * config.NUM_LABELS,config.noise_dim),device=config.device)
                fake_out = G(noise_test,test_hot_label)
                save_image(fake_out, f'images/{batch_iter}.png', nrow=5, normalize=True)


if __name__ == '__main__':
    config = model.Config()
    # 加载数据集
    train_iter = load_file(config)
    # 载入模型
    model_d = model.ModelD().to(config.device)
    model_g = model.ModelG(config.noise_dim).to(config.device)
    # 训练
    train(train_iter, model_g, model_d, config)
