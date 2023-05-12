import os
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

class Config(object):
    def __init__(self,path,dataset,embedding):
        self.train_dir = os.path.join(dataset,'NEWS/train.txt')
        self.dev_dir = os.path.join(dataset,'NEWS/dev.txt')
        self.test_dir = os.path.join(dataset,'NEWS/test.txt')
        self.vocab_dir = os.path.join(dataset,'NEWS/vocab.pkl')
        self.save_model_dir = os.path.join(path,'THUCNews/saved_dict/TextCNN.ckpt')

        self.max_size = 10000
        self.limit_seq = 1
        self.batch_size = 128
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pad_size = 32
        self.pretrain_embedding = torch.tensor(np.load(os.path.join
                                  (dataset,'NEWS/embedding_SougouNews.npz'))['embeddings'],
                                  dtype=torch.float32) if embedding != 'random' else None
        self.embed_size = 300
        self.vocab_size = 0
        self.filter_size = [2,3,4]
        self.filter_num = 256
        self.num_classes = 10
        self.dropout = 0.5
        self.num_epochs = 20
        self.lr = 0.001
        self.judge_batch = 1000

'''
# 利用二维卷积来构建模型
class Model(nn.Module):
    def __init__(self,config):
        super(Model, self).__init__()
        if config.pretrain_embedding is None:
            self.embedding = nn.Embedding(config.vocab_size,config.embed_size)
        else:
            self.embedding = nn.Embedding.from_pretrained(config.pretrain_embedding,freeze=False) # 参与到训练中
        self.conv = nn.ModuleList([nn.Conv2d(in_channels=1,out_channels=config.filter_num,
                                             kernel_size=(i,config.embed_size)) for i in config.filter_size])
        self.fc = nn.Linear(config.filter_num * len(config.filter_size),config.num_classes)
        self.dropout = nn.Dropout(config.dropout)

    def pool_layer(self,out,blk):
        out = F.relu(blk(out)).squeeze(3) # [batch,out_channel,step-kernel_size+1]
        out = F.max_pool1d(out,out.shape[2]).squeeze(2) # [batch,out_channel,1]
        return out

    def forward(self,x): # 通过数据集得到的x是一个元组，包含序列和实际的长度
        out = self.embedding(x[0]).unsqueeze(1) # [batch,1,seq,embed_size]
        out = torch.cat([self.pool_layer(out,blk) for blk in self.conv],dim=1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
'''

# 利用一维卷积来构建模型(一维的卷积的运算速度更快)（参数需要调整，出现了过拟合）
class Model(nn.Module):
    def __init__(self,config):
        super(Model, self).__init__()
        if config.pretrain_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(config.pretrain_embedding,freeze=False)
        else:
            self.embedding = nn.Embedding(config.vocab_size,config.embed_size)
        self.conv = nn.ModuleList([nn.Conv1d(in_channels=config.embed_size,
                                             out_channels=config.filter_num,kernel_size=i) for i in config.filter_size])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.filter_num * len(config.filter_size), config.num_classes)

    def pool_layer(self,out,blk):
        out = blk(out) # [batch,filter_num,step-kernel+1]
        out = F.relu(out)
        out = F.max_pool1d(out,out.shape[2]).squeeze(2) # [batch,filter_num]
        return out

    def forward(self,x):
        out = self.embedding(x[0]).permute(0,2,1) # [batch,embed_size,step]
        out = torch.cat([self.pool_layer(out,blk) for blk in self.conv],dim=1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
