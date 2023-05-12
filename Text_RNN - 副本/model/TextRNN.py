import torch
import numpy as np
from torch import nn
import os

class Config(object):
    def __init__(self,dataset,embedding,work_path=None):
        self.train_path = os.path.join(dataset,'data/train.txt')
        self.dev_path = os.path.join(dataset,'data/dev.txt')
        self.test_path = os.path.join(dataset,'data/test.txt')
        if work_path is not None:
            self.vocab_path = os.path.join(work_path,'NEWS/data/vocab.pkl')
            self.save_model_path = os.path.join(work_path,'NEWS/state_dict/model1.ckpt')
        self.classes_num = len(open(os.path.join(dataset,'data/class.txt')).readlines())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pretrain_embedding = torch.tensor(np.load(os.path.join(dataset,f'data/{embedding}'))['embeddings'],
                                               dtype=torch.float32) if embedding != 'random' else None
        self.embed_size = self.pretrain_embedding.shape[1] if self.pretrain_embedding is not None else 300
        self.vocab_size = 0
        self.num_hidden = 128
        self.num_layers = 2
        self.dropout = 0.5
        self.pad_size = 32
        self.use_word = False
        self.batch_size = 32
        self.lr = 1e-3
        self.sum_epoch = 20
        self.requirement = 1000

class Model(nn.Module):
    def __init__(self,config):
        super(Model, self).__init__()
        print(config.embed_size)
        if config.pretrain_embedding is None:
            self.embedding = nn.Embedding(config.vocab_size,config.embed_size)
        else:
            self.embedding = nn.Embedding.from_pretrained(config.pretrain_embedding,freeze=False) # 已验证embed参加训练
        self.lstm = nn.LSTM(config.embed_size,config.num_hidden,
                            num_layers=config.num_layers,bidirectional=True,dropout=config.dropout)
        if self.lstm.bidirectional:
            self.direction = 2
        else:
            self.direction = 1
        self.num_layers = config.num_layers
        self.num_hidden = config.num_hidden
        self.fc = nn.Linear(config.num_hidden * self.direction,config.classes_num)

    def init_state(self,batch,device):
        return (torch.randn(size=[self.direction * self.num_layers, batch, self.num_hidden],dtype=torch.float32,device=device),
                torch.randn(size=[self.direction * self.num_layers, batch, self.num_hidden], dtype=torch.float32,
                            device=device))

    def forward(self,x):
        out = self.embedding(x[0]).permute(1,0,2) # [seq,batch,embed_size]
        # 尽量不去使用out,state = self.lstm(out,state)，其中state是自己定义的，会让反向传播时出现问题
        out,state = self.lstm(out) # out = [seq,batch,direction * num_hidden]
        out = out.permute(1,0,2) # [batch,seq,direction * num_hidden]
        out = self.fc(out[:,-1,:]) # [batch,num_classes]
        return out,state

'''
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.pretrain_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(config.pretrain_embedding,freeze=False) # 已验证embed参加训练
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.embed_size,config.num_hidden,num_layers=config.num_layers,
                            bidirectional=True, dropout=config.dropout)
        self.fc = nn.Linear(config.num_hidden * 2, config.classes_num)
        self.direction = 2 if self.lstm.bidirectional else 1
        self.num_layers = config.num_layers
        self.num_hidden = config.num_hidden

    def forward(self, x):
        x, _ = x
        out = self.embedding(x).permute(1,0,2)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        out, state = self.lstm(out)
        out = out.permute(1,0,2)
        out = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state
        return out,state

    def init_state(self,batch,device): # 自己创建的初始状态会产生问题，和初始产生的不一样
        return (torch.randn(size=[self.direction * self.num_layers, batch, self.num_hidden],dtype=torch.float32,device=device),
                torch.randn(size=[self.direction * self.num_layers, batch, self.num_hidden], dtype=torch.float32,
                            device=device))
'''

if __name__ == '__main__':
    data_dir = 'D:\\python\\pytorch作业\\nlp实战练习\\Text_CNN\\textcnn\\THUCNews'
    embedding = 'pre_train'
    config = Config(data_dir,embedding)
    model = Model(config)
    model.to(config.device)
    x = torch.tensor([[1,2,3],[4,5,6]],dtype=torch.long,device=config.device)
    x = (x,1)
    state = model.init_state(2,config.device)
    out = model(x,state)
    print(out.shape)

