# 用于计算FID
from scipy import linalg
import torch
from torch import nn
import numpy as np
import torchvision.transforms as TF
from torch.utils import data
import pathlib
from pytorch_fid.inception import InceptionV3
from PIL import Image




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
    path = pathlib.Path(config.input_path)
    files = sorted(path.glob('*.jpg'))
    if config.batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        config.batch_size = len(files)
    # 利用ToTensor将其归一化同时修正通道的位置
    dataset = ImagePathDataset(files,
                transforms=TF.Compose([TF.Resize((config.image_size,config.image_size)),TF.ToTensor(),TF.Normalize(mean=[0.5] * 3, std=[0.5] * 3)]))
    if mode == 0:  # 用来加载训练的数据集
        dataloader = data.DataLoader(dataset,batch_size=config.batch_size,shuffle=True,drop_last=True)
    else: # 用来抽取特征
        dataloader = data.DataLoader(dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)
    return dataloader,files