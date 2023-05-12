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
        if config.pretrain_embedding is None:
            self.embedding = nn.Embedding(config.vocab_size,config.num_hidden)
        else:
            self.embedding = nn.Embedding.from_pretrained(config.pretrain_embedding,freeze=False)
        self.lstm = nn.LSTM(config.embed_size,config.num_hidden,num_layers=config.num_layers,
                            bidirectional=True,dropout=config.dropout)
        self.fc = nn.Linear(config.num_hidden * 2 + config.embed_size, config.classes_num)
        self.pool = nn.MaxPool1d(kernel_size=config.pad_size) # 只针对最后一个维度
    def forward(self,x):
        inputs, _ = x # [batch,seq]
        embed = self.embedding(inputs).permute(1,0,2) # [seq,batch,embed_size]
        out, _ = self.lstm(embed) # out = [seq,batch,num_hidden * 2]
        out = torch.cat((embed,out),dim=2) # [seq,batch,num_hidden * 2 + embed_size]
        out = out.permute(1,0,2) # [batch,seq,num_hidden * 2 + embed_size]
        out = out.permute(0,2,1) # [batch,num_hidden * 2 + embed_size,seq]
        out = self.pool(out).squeeze(2) # [batch,num_hidden * 2 + embed_size]
        out = self.fc(out)
        return out # [batch,num_classes]



if __name__ == '__main__':
    x = torch.arange(40,dtype=torch.float32).reshape(4,2,5) # [seq=4,batch=2,embed_size=5]
    pool = nn.MaxPool1d(kernel_size=5)
    print(pool(x).shape)
