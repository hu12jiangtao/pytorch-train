# 在一个判别的神经网络D中，其中包含含的叶子节点有:输入的图片和自己设置的每一层的权重
# 叶子节点的满足要求require_grad=Flase的节点(不进行训练) 或者 require_grad=True但是是自己设置的例如每一层的权重(训练的参数)

# WGAN-GP在WGAN上的改进
# 在improve-WGAN中针对于D 满足1-Lipschitz 等价于 ||grad(D(x)|| <= 1(对于D中的所有参数) ==> 针对生成的分数对图片的每一个像素点求导
# 因此其中的一个思想为对于||grad(D(x)|| > 1 的地方添加一个惩罚项，
# 此时的目标函数为变为V(G,D)=max[Ex~data(D(x)) - Ex~pG(D(x)) - lambda * ∫x max(0,||grad(D(x)|| - 1) dx]
# 但是这样会存在一个问题:此时针对的是输入判别器的图像的所有的像素点，因此会造成巨大的计算量,因此将所有输入D的图片的分布转换成 输入D的真图片的分布和假图像分布之间的分布X_penalty
# 此时X_penalty相当于整个X分布中的一小部分，相当于减少了计算量
# 同时∫x max(0,||grad(D(x)|| - 1) dx 需要进行修正，变为lambda * (||grad(D(x)|| - 1)**2 ，期望x的梯度越接近1越好(实验做出来的)

from torchvision.utils import save_image
import torch
import model
from torch import nn
import numpy as np
import utils
import pickle


def gradient_penalty(critic, real, fake, device="cpu"):
    batch_size,channel,W,H = real.shape
    alpha = torch.rand(size=(batch_size,1,1,1),device=device).repeat((1,channel,W,H))
    # sample_image代表着在X_penalty中采样的点
    sample_image = alpha * real + (1 - alpha) * fake
    sample_out = critic(sample_image)
    # torch.autograd.grad是针对于张量对张量的求导，grad_outputs应当和outputs形状相同，用来分配每一项在求导时占据的权重，其中的0代表着权重参数的信息存放在[0]中
    # gradient中只存放着inputs中每个像素对应的梯度
    # 虽然此时的D的参数和输入都是叶子节点(requires_grad=True,但是都是用户自己设计)，
    # 但是torch.autograd.grad是只将inputs作为未知参数，D的网络参数作为已知量进行求导，因此输出应当和输入的形状相同
    # inputs必须requires_grad=True,torch.autograd.grad此时的输出的形状应当和输入的sample_image相同

    # 当retain_graph=False,create_graph=False的情况下程序执行了:D的输出对D的输入进行求导后自动将计算图给释放了
    # 正常的情况下应该建立整个图，首先根据前向传播得到的out_real、out_fake创建部分计算图，
    # 之后需要提供获得正则项的计算图，就是torch.autograd.grad中所创建的计算图，因此在torch.autograd.grad中create_graph=True
    # 因此当运行完torch.autograd.grad不应该将计算图给释放，就是retain_graph=True
    # 在创建完整个完整的图后应当进行反向传播以及梯度更新，正则项中更新的是生成的图片(将其往真实图片的分布中进行更新)【重点！！】
    grad = torch.autograd.grad(inputs=sample_image,outputs=sample_out,
                               grad_outputs=torch.ones_like(sample_out,device=device),
                               retain_graph=True,create_graph=True)[0] # [batch,channel,w,h]
    grad = grad.reshape(grad.shape[0], -1)
    grad_norm = grad.norm(2,dim=1) # [batch,]
    grad_penalty = torch.mean((grad_norm - 1)**2)
    return grad_penalty # [1,]

def set_grad_require(net,require):
    for param in net.parameters():
        param.requires_grad = require

def train(data_iter, G, D, config):
    trainer_G = torch.optim.Adam(G.parameters(),lr=config.lr_G,betas=(0.0, 0.9))
    trainer_D = torch.optim.Adam(D.parameters(),lr=config.lr_D,betas=(0.0, 0.9))
    step = 0
    fid_value_lst = []
    for epochs in range(config.num_epochs):
        for x in data_iter:
            G.train()
            D.train()
            x = x.to(config.device)
            # 对判别器进行训练
            set_grad_require(D, require=True)
            for _ in range(config.CRITIC_ITERATIONS):
                noise = torch.randn(size=(config.batch_size,config.channel_noise,1,1),device=config.device)
                fake = G(noise)
                real = x
                gp = gradient_penalty(D, real, fake, device=config.device)
                out_fake = D(fake.detach()).reshape(-1)
                out_real = D(real).reshape(-1)
                # 此时的D_loss加入了惩罚项gp，其中gp与D的输出对D的输入进行求导
                # 存在正则项的loss函数:▽ D(x)先使用autograd.grad()函数求D(x)对x的导数，将retain_graph，create_graph都设置成True添加导数计算图并保存计算图，
                # 再加上loss的其他部分，最后使用backward()函数即可
                D_loss = -(torch.mean(out_real) - torch.mean(out_fake) - gp)
                trainer_D.zero_grad()
                D_loss.backward()
                trainer_D.step()
            # 对生成器进行训练
            set_grad_require(D, require=False)
            noise = torch.randn(size=(config.batch_size, config.channel_noise, 1, 1), device=config.device)
            fake = G(noise)
            out_fake = D(fake).reshape(-1)
            G_loss = -torch.mean(out_fake)
            trainer_G.zero_grad()
            G_loss.backward(retain_graph=True)
            trainer_G.step()
            if step % 500 == 0:
                G.eval()
                D.eval()
                print(f'D_loss:{D_loss:.4f}  G_loss:{G_loss:.4f}')
                with torch.no_grad():
                    noise = torch.randn(size=(config.batch_size, config.channel_noise, 1, 1), device=config.device)
                    fake = G(noise)
                # fid 的计算
                with torch.no_grad():
                    pred = get_feature_model(fake)[0]  # [batch,dim,1,1]
                if pred.shape[2] != 1 or pred.shape[3] != 1:
                    pred = nn.AdaptiveAvgPool2d((1, 1))(pred)
                pred = pred.squeeze(3).squeeze(2).cpu().numpy()  # [batch,dim]
                mu_G = np.mean(pred, axis=0)  # mu = [dim(2048),]
                sigma_G = np.cov(pred, rowvar=False)  # 求解这个矩阵的协方差 sigma=[2048,2048]
                fid_value = utils.calculate_frechet_distance(mu, sigma, mu_G, sigma_G, eps=1e-6)
                print(f'fid_value:{fid_value:.3f}')
                fid_value_lst.append(fid_value)
                save_image(fake[:10],f'images/{step}.png',nrow=5,normalize=True)
            step += 1
    return fid_value_lst


if __name__ == '__main__':
    config = model.Config()
    # 计算整体图像的FID
    # 获取抽取模型
    get_feature_model = utils.load_model(config)
    # 加载数据集并且求解原始数据集的均值和协方差
    mu, sigma = utils.get_mean_std(config, get_feature_model)

    # 导入图片
    data_iter,_ = utils.load_data(config, mode=0)
    # 导入生成器和判别器
    discriminator = model.Discriminator(config.channel_image,config.feature_g).to(config.device)
    generator = model.Generator(config.channel_noise, config.channel_image, config.feature_g).to(config.device)
    discriminator.apply(model.initialize_weights)
    generator.apply(model.initialize_weights)
    # 进行训练
    fid_value_lst = train(data_iter, generator, discriminator, config)
    # save
    with open(config.path_lst, 'wb') as f:
        pickle.dump(fid_value_lst, f)