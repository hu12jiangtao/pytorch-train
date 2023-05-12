from torch import nn
import torch
import os
import numpy as np
from torch.nn import functional as F

class Config(object):  # 配置训练时的各种参数
    def __init__(self,dataset,embedding):
        self.model_name = 'TextCNN'
        self.train_path = os.path.join(dataset,'data/train.txt')
        self.test_path = os.path.join(dataset,'data/test.txt')
        self.dev_path = os.path.join(dataset,'data/dev.txt')
        self.vocab_path = os.path.join(dataset,'data/vocab.pkl')
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果

        self.embedding_pretrained = torch.tensor(np.load(os.path.join(dataset,f'data/{embedding}'))["embeddings"],
                                                 dtype=torch.float32) if embedding != 'random' else None
        self.embed_size = self.embedding_pretrained.shape[1] if self.embedding_pretrained is not None else 300

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_list = [i.strip() for i in open(dataset + '/data/class.txt','r',encoding='utf-8').readlines()]
        self.num_classes = len(self.class_list)

        self.n_vocab = 0
        self.pad_size = 32
        self.batch_size = 128
        self.filter_sizes = (2, 3, 4)
        self.num_filters = 256
        self.dropout = 0.5
        self.learning_rate = 0.001
        self.num_epochs = 20
        self.require_improvement = 1000


class Model(nn.Module):
    def __init__(self,config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            # freeze=True的情况下在训练过程中embedding层的参数不进行训练，由于此时是迁移学习因此会将freeze设置为False
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained,freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab,config.embed_size)
        # 其目标是对seq进行缩减，因此
        # Con1d相当于batch批量的通道数为embed_size的[seq,]向量 和 个数为out_channels的通道数为embed_size过滤器[kernel_size,]向量进行卷积
        # input=[batch,seq,embed_size]->input=input.permute(0,2,1)=[batch,embed_size,seq]
        # ->conv=(in_channel=embed_size,out_channel=filter_nums,kernel)->output=conv(input)=[batch,filers_nums,seq-kernel_size+1]

        # 等价二维卷积input=[batch,1,seq,embed_size]->conv=(in_channel=1,out_channel=filter_nums,kernel=(kernel,kernel_size))
        # ->output=[batch,filers_nums,seq-kernel_size+1,1]
        self.conv = nn.ModuleList([nn.Conv2d(in_channels=1,out_channels=config.num_filters,kernel_size=(i,config.embed_size))
                                   for i in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, out, blk):
        out = F.relu(blk(out)).squeeze(3) # [batch,filter_num,seq]
        out = F.max_pool1d(out,out.shape[2]).squeeze(2) # [batch,filter_num]
        return out

    def forward(self,x): # 此时的输入为一个元组，包含着(输入模型部分，实际长度)
        out = self.embedding(x[0]).unsqueeze(1) # [batch,1,seq,embed_size]
        out = torch.cat(([self.conv_and_pool(out, blk) for blk in self.conv]),dim=1) # [batch,filter_num_sum]
        out = self.dropout(out)
        out = self.fc(out)
        return out # [batch,num_class]





