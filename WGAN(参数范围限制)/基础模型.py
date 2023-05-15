from scipy import linalg
import torch
from torch import nn
import numpy as np
import torchvision.transforms as TF
from torch.utils import data
import os
import pathlib
import pickle
from pytorch_fid.inception import InceptionV3
from PIL import Image
from torchvision.utils import save_image

'''
# 模式坍塌的解决方式
# 1.利用BCELoss来替换MSELoss（针对普通的GAN）
# 2.鉴别器中使用 LeakReLU() 激活函数。
# 3.提供更多的输入种子来优化，比如可以尝试采用 100 个输入节点，每个节点都是随机值（生成器上）
# 4.尽量生成器和判别器应当对称（WGNAN中论文中提到不需要再考虑）
# 利用WGAN来跳出模式崩溃(添加梯度惩罚)
'''
# 总结(重要!!!)
# WGAN针对于GAN网络来说需要修正的地方:在判别器中去掉最后的sigmoid函数(非线性)，将其变为线性的；另一个是修改目标函数；最后是需要加上param clipping


# WGAN 针对是判别器(出现模式崩溃的原因是 判别器太过于超前于生成器(判别器的损失到达接近0，此时相当于判别器已经训练完成之后单独训练生成器、而不是判别器和生成器交替迭代))
# 超前的意思是对于更新后的生成器，原来的判别器仍然适用

# 模式崩溃存在两种现象
# 1.模式崩塌:生成数据的分布和原来的数据分布之间完全不同
#          直观理解:对于G输入的生成图像，判别器D认为几张图片是真实的，因此这个的生成器就不进行训练，只生成那几张D认为真实的图片
# 2. 梯度消失原因
#   具体解释:
#          一种解释方式:
#          判别器D是一个二分类的问题，最后对判别器的输出进行sigmoid处理，D训练完成后 生成的样本点在接近0的附近，真实样本在1的附近，
#          而生成器的作用是将生成的点(输出分数为0)根据判别器D的移动到输出分数为1的区域，但是由于sigmoid函数在0的情况下梯度很小因此会出现训练不动的情况
#          另一种解释方式:
#          BCELoss的推导过程利用到了JS散度(非黑即白的，不能知道两个分布有多不相似)，两个分布离得较近的情况下JS值为log2；两个分布离得较远的情况下JS值为log2，只有当两个分布重合的情况下才会使JS=0
#          这种情况会对training会造成影响、生成器training的本质是使divergence（分布）相差最小(JS散度最小),但是由于只要是分布不相同的两个分布JS值都是log2
#          因此无法将两个分布从分布距离较远的情况下移动到分布较近的情况下，因为两个分布的JS散度的值都是log2
#   过去的解决方法:将判别器的学习率降低使得D不被训练的太好(太小无法判断对生成图片判别，太小则会梯度消失)
#   解决方法:
#   1.将判别器的sigmoid函数改为linear,此时将D右判别任务(classification problem)变成了回归(regression),此时生成的点经过D的输出越接近0越好、真实点越接近1越好
#   2.换一种目标函数(WGAN-此时D的输出应当是线性的，最后没有sigmoid函数):换一种目标函数,将原来JS换成Earth Movers’ Distance(给了两个分布一个相似的分数，越相似分数越小)
#     EMS的解释:给定分布P/Q，将P中的所有的点X_p位置经过变换得到X_q的位置，这种方法称为推土方法f，此时存在很多种推土方法
#             举例:二维的分布中此时 P的1处有3个点，2处有2个点，3处有1个点；P的1处有0个点，2处有4个点，3处有2个点 此时可以将P的1处3点给2，2的1个点给3 或者也可以 P的1处2点给2，1的1个点给3等等
#                 此时的距离Distance(sigma为求和符号): sigma(xq,xp)f(xq,xp)*|xq - xp| 上述第二种就是 (2-1)*2 + (3-1)*1
#                 而EMD则是去最优方案的f使得D最小
#     根据EMD可以推断出损失函数:V(G,D)=max(Ex~data(D(x)) - Ex~pG(D(x))) -> 要求训练出一个判别器参数使生成数据D的输出越小越好、真实数据D的输出越大越好
#     此时的生成器的损失函数应当为max(Ex~pG(D(x)))
#     同时对于判别器D来说必须满足1-Lipschitz(足够的平滑),原因是若不是足够的平滑的情况下 x~data对应的部分越小越好直至-inf，x~pG对应的部分越大越好直至inf
#     这样就这个值就不会收敛了(V(G,D)是一个inf的值，会不断变大)，因此要加上D满足1-Lipschitz条件(||D(x1) - D(x2)|| < ||x1 - x2||)
#     如何让D被contrain(限制)呢? WGAN中是对D的参数进行限制在[-c,c]之间(当参数小于-c时变为-c，当参数大于c时变为c)




class Generator(nn.Module):
    """
    input (N, in_dim)
    output (N, 3, 64, 64)
    """
    def __init__(self, in_dim, dim=64):
        super(Generator, self).__init__()

        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 5, 2,
                                   padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2))
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.LeakyReLU(0.2))
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 4),
            dconv_bn_relu(dim * 4, dim * 2),
            dconv_bn_relu(dim * 2, dim),
            nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1),
            nn.Tanh())

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y


class Discriminator(nn.Module):
    """
    input (N, 3, 64, 64)
    output (N, )
    """
    def __init__(self, in_dim, dim=64):
        super(Discriminator, self).__init__()

        def conv_bn_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2))
        self.ls = nn.Sequential(
            nn.Conv2d(in_dim, dim, 5, 2, 2), nn.LeakyReLU(0.2),
            conv_bn_lrelu(dim, dim * 2),
            conv_bn_lrelu(dim * 2, dim * 4),
            conv_bn_lrelu(dim * 4, dim * 8),
            nn.Conv2d(dim * 8, 1, 4),
            nn.Sigmoid()  # nn.Sigmoid()应该拿掉的，在WGAN中中D的输出应当是线性的
            )

    def forward(self, x):
        y = self.ls(x)
        y = y.view(-1)
        return y  # [batch,]


# 获取原始数据集的分布情况
def load_model(config):  # fid所利用到的抽取特征的模型
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[config.dims] # 通过抽取特征的维度判断需要InceptionV3中多少个block
    model = InceptionV3([block_idx]).to(config.device) # 提取出对应的block作为抽取特征的网络
    return model

class ImagePathDataset(data.Dataset):
    def __init__(self,files,transforms=None):
        super(ImagePathDataset, self).__init__()
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        image_path = self.files[i] # 给出对应图片的路径
        image_data = Image.open(image_path).convert('RGB')
        if self.transforms is not None:
            image_data = self.transforms(image_data)
        return image_data

def get_activations(files, dataloader, model, config):  # 获取所有图片的特征
    model.eval()
    pred_arr = np.empty((len(files), config.dims)) # 用来存放所有的图片的抽取特征(一张图片为一行2048的向量)
    start_idx = 0
    for x in dataloader:
        x = x.to(config.device) # [batch,3,64，64]
        with torch.no_grad():
            pred = model(x)[0]  # [batch,dim,1,1]
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = nn.AdaptiveAvgPool2d((1,1))(pred)
        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        pred_arr[start_idx:start_idx + config.batch_size] = pred
        start_idx += config.batch_size
    return pred_arr # 存放在cpu，numpy类型的一个文件夹下所有图片的特征

def get_mean_std(config, model):
    dataloader,files = load_data(config,mode=1)
    act = get_activations(files, dataloader, model, config)
    mu = np.mean(act,axis=0) # mu = [dim(2048),]
    sigma = np.cov(act,rowvar=False) # 求解这个矩阵的协方差 sigma=[2048,2048],解释见1.py中案例,(rowvar=False代表每一行之间的方差)
    return mu,sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    # X_1 ~ N(mu_1, C_1); X_2 ~ N(mu_2, C_2)
    # 公式:d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'
    diff = mu1 - mu2
    covmean,_ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot((sigma2 + offset)))
    if np.iscomplexobj(covmean): # 判断是否含有虚数，含有虚数时为真
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
        # .imag代表取虚数部分，diagonal代表取对角线上元素,np.allclose用来判断列表中元素是否相同，允许误差1e-3
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real # 等于原来的实部
    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

# 加在训练集的数据
def load_data(config, mode):
    path = pathlib.Path(config.path)
    files = sorted(path.glob('*.jpg'))
    if config.batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        config.batch_size = len(files)
    # 利用ToTensor将其归一化同时修正通道的位置
    dataset = ImagePathDataset(files,
                transforms=TF.Compose([TF.Resize((64,64)),TF.ToTensor(),TF.Normalize(mean=[0.5] * 3, std=[0.5] * 3)]))
    if mode == 0:  # 用来加载训练的数据集
        dataloader = data.DataLoader(dataset,batch_size=config.batch_size,shuffle=True)
    else: # 用来抽取特征
        dataloader = data.DataLoader(dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)
    return dataloader,files


def set_requires_grad(net,requires_grad=False):
    for param in net.parameters():
        param.requires_grad = requires_grad


def train_Wgan(train_iter, G, D, config):
    trainer_G = torch.optim.Adam(G.parameters(),lr=0.001,betas=(0.5,0.999))
    trainer_D = torch.optim.Adam(D.parameters(), lr=0.001, betas=(0.5, 0.999))
    step = 0
    fid_value_lst = []
    for epoch in range(config.num_epochs):
        for X in train_iter:
            G.train()
            D.train()
            X = X.to(config.device)
            # 进行判别器的训练(此时并没有关闭)
            set_requires_grad(D, requires_grad=True)
            for _ in range(5):
                noise = torch.randn((X.shape[0], config.z_dim), device=config.device)
                fake = G(noise)
                real = X
                fake_D = D(fake.detach())
                real_D = D(real)
                loss_D = -(torch.mean(real_D) - torch.mean(fake_D))
                trainer_D.zero_grad()
                loss_D.backward()
                trainer_D.step()
                for p in D.parameters():
                    p.data.clamp_(-0.02,0.02)

            # 进行生成器的训练
            set_requires_grad(D, requires_grad=False)
            noise = torch.randn((X.shape[0], config.z_dim), device=config.device)
            gen_fake = G(noise)
            gen_fake_D = D(gen_fake)
            loss_G = -(torch.mean(gen_fake_D))
            trainer_G.zero_grad()
            loss_G.backward()
            trainer_G.step()

            if step % 50 == 0:
                G.eval()
                D.eval()
                print(f'loss_D:{loss_D:.4f}  loss_G:{loss_G:.4f}')
                with torch.no_grad():
                    noise = torch.randn((X.shape[0], config.z_dim), device=config.device)
                    gen_fake = G(noise)
                # fid 的计算
                with torch.no_grad():
                    pred = get_feature_model(gen_fake)[0]  # [batch,dim,1,1]
                if pred.shape[2] != 1 or pred.shape[3] != 1:
                    pred = nn.AdaptiveAvgPool2d((1, 1))(pred)
                pred = pred.squeeze(3).squeeze(2).cpu().numpy()  # [batch,dim]
                mu_G = np.mean(pred, axis=0)  # mu = [dim(2048),]
                sigma_G = np.cov(pred, rowvar=False)  # 求解这个矩阵的协方差 sigma=[2048,2048]
                fid_value = calculate_frechet_distance(mu, sigma, mu_G, sigma_G, eps=1e-6)
                print(f'fid_value:{fid_value:.3f}')
                fid_value_lst.append(fid_value)
                save_image(gen_fake[:10],f'face_image/{step}.png',nrow=10,normalize=True)
            step += 1


class Config(object):
    def __init__(self):
        self.batch_size = 20
        self.path = 'faces'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dims = 2048
        self.z_dim = 100
        self.in_channel = 3
        self.num_epochs = 1000
        self.path_lst = os.path.join('out', 'fid_lst.pkl')
        self.lambda_gp = 10

if __name__ == '__main__':
    config = Config()
    if not os.path.exists(config.path_lst):
        # 获取抽取模型
        get_feature_model = load_model(config)
        # 加载数据集并且求解原始数据集的均值和协方差
        mu,sigma = get_mean_std(config, get_feature_model)
        # 获取训练集的数据
        train_iter, _ = load_data(config, mode=0)
        # 生成器、判别器的定义
        G = Generator(in_dim=config.z_dim)
        D = Discriminator(config.in_channel)
        G.to(config.device)
        D.to(config.device)
        # 利用WGAN进行训练
        train_Wgan(train_iter, G, D, config)
    '''
        # 对其进行训练
        fid_value_lst = train(train_iter, G, D, get_feature_model, mu, sigma, config)
        # save
        with open(config.path_lst, 'wb') as f:
            pickle.dump(fid_value_lst, f)
        print(fid_value_lst)

    if os.path.exists(config.path_lst):
        fid_value_lst = pickle.load(open(config.path_lst,'rb'))
        print(fid_value_lst)
    '''







