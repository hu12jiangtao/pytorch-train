# 此时需要进行可视化(利用训练好的MoCo模型中的encode_q将一张图片转换为一个特征)
# MoCo模型相当于一个无监督的预训练模型
import random
import matplotlib.pyplot as plt
import numpy as np
import setuptools.sandbox
import torch
import torchvision
from torch import nn
import Models
import os
import pickle
import main

class Config(object):
    def __init__(self):
        self.save_model_path = 'checkpoint_out/wide_resnet'
        self.save_feature_path = 'checkpoint_out/features.pkl'
        self.device = torch.device('cuda')

if __name__ == '__main__':
    config = Config()
    label_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] # 十个类
    # 导入数据集，此时的目标是对于一张图片来说，转换后特征最相近的图片和特征最不相近的图片的可视化
    # 一个良好的模型应当对于相同类别的不同图片是相近的，不同类别的图片是特征相差较大的(可以用于迁移后在下游任务上进行微调)

    # 导入数据集(此时无需进行数据增强)
    test_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                                                                      std=(0.2471, 0.2435, 0.2616))])
    train_dataset = torchvision.datasets.cifar.CIFAR10(root='D:\\python\\pytorch作业\\all_data\\cifar10\\data',
                                                       transform=None,download=False,train=True) # 由于之后需要显示图片因此此时并没有数据增强
    # 导入模型
    model_q,_ = main.get_model('wide_resnet', config) # 此时只是导入了模型的初始参数，并没有导入模型已经训练好的参数
    model_q.load_state_dict(torch.load(config.save_model_path)['model_q'])
    model_q.eval()
    # 开始进行测试
    labels = []
    features = []

    if os.path.exists(config.save_feature_path):
        with open(config.save_feature_path,'rb') as f:
            all_data = pickle.load(f)
            features = all_data[0]
            labels = all_data[1]
    else:
        with torch.no_grad():
            for x, y in train_dataset:
                labels.append(label_list[y])
                x = test_transform(x).to(config.device) # [3,32,32]
                x = x.unsqueeze(0) # [1,3,32,32]
                feature = model_q(x) # [1,128]
                features.append(feature.cpu().numpy()[0]) # 每一个元素为128维的向量,长度为训练集的长度
            with open(config.save_feature_path,'wb') as f:
                pickle.dump([features,labels],f)
    # 随机选择出一张图片，取出最相近的图片和最不相近的图片
    print(len(labels))
    pos_idx = random.choice([i for i in range(len(features))])
    pos_sample, pos_label = features[pos_idx], labels[pos_idx]

    features = np.stack(features,axis=0) # [batch,128]
    pos_sample = torch.tensor(pos_sample,device=config.device).unsqueeze(0) # [1,128]
    candidate_sample = torch.tensor(features,device=config.device) # [batch,128]
    # 此时y_hat就是两个特征向量的点积(当点积的值越大，说明两个特征越接近)
    result = torch.matmul(pos_sample,candidate_sample.permute(1,0)).squeeze() # [batch, ]
    sim_with_label = [(index,dot) for index,dot in enumerate(result)] # index代表着candidate_sample中的第index张图片
    sim_with_label = sorted(sim_with_label,key=lambda x:x[1]) # 按点积的从小到大的值进行排列
    # 进行可视化
    col_num = 5
    f = plt.figure()
    pos_img,_ = train_dataset[pos_idx]
    f.add_subplot(3,5,1) # 将一张图片分为(3,5)的坐标，并且将之后的图片显示在位置1的地方
    plt.imshow(pos_img)
    plt.title(pos_label)
    # 显示最相近的5张图片
    positive = sim_with_label[-col_num:]
    for idx,(index,dot) in enumerate(positive):
        positive_img, _ = train_dataset[index]
        label = labels[index]
        f.add_subplot(3, 5, 6 + idx)
        plt.imshow(positive_img)
        plt.title(f'{label}_{dot:1.2f}')
    # 显示最不相近的5张图片
    neg = sim_with_label[:col_num]
    for idx, (index,dot) in enumerate(neg):
        neg_img, _ = train_dataset[index]
        label = labels[index]
        f.add_subplot(3, 5, 11 + idx)
        plt.imshow(neg_img)
        plt.title(f'{label}_{dot:1.2f}')
    plt.show()













