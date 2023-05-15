import torch
from torch import nn
from torch.nn import functional as F

class Config(object):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 128
        self.each_test_num = 5
        self.NUM_LABELS = 10
        self.noise_dim = 100
        self.num_epochs = 25


class ModelD(nn.Module):
    def __init__(self):
        super(ModelD, self).__init__()
        self.part = nn.Sequential(self.blk(1,32,5,1,2), self.blk(32,64,5,1,2))
        self.fc1 = nn.Linear(10,1000)
        self.relu1 = nn.LeakyReLU(0.1)
        self.relu2 = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(64 * 28 * 28 + 1000, 1024)
        self.fc3 = nn.Linear(1024,1)
        self.sigmoid = nn.Sigmoid()

    def blk(self,in_channel,out_channel,k,s,p):
        return nn.Sequential(nn.Conv2d(in_channel,out_channel,kernel_size=k,stride=s,padding=p),
                             nn.BatchNorm2d(out_channel),nn.LeakyReLU(0.1))

    def forward(self,x,labels): # x=[batch,1,28,28] labels=[batch,10]
        out1 = self.part(x).reshape(x.shape[0],-1)
        out2 = self.relu1(self.fc1(labels))
        out = torch.cat([out1,out2],dim=1)
        out = self.fc3(self.relu2(self.fc2(out)))
        return self.sigmoid(out) # [batch,1]

class ModelG(nn.Module):
    def __init__(self,z_dim):
        super(ModelG, self).__init__()
        self.fc1 = nn.Linear(10,1000)
        self.fc2 = nn.Linear(1000 + z_dim, 64 * 28 * 28)
        self.part = nn.Sequential(self.blk(64,32,5,1,2),self.blk(32,1,5,1,2))
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU(0.1)

    def blk(self,in_channel,out_channel,k,s,p):
        return nn.Sequential(nn.BatchNorm2d(in_channel),nn.LeakyReLU(0.1),
                             nn.ConvTranspose2d(in_channel,out_channel,kernel_size=k,stride=s,padding=p))

    def forward(self,x,labels): # x=[batch,100] ,label=[batch,10]
        out1 = self.fc1(labels)
        out1 = self.relu(out1)
        out = torch.cat([x,out1],dim=1)
        out = self.fc2(out).reshape(-1, 64, 28, 28)
        out = self.part(out)
        return self.sigmoid(out)





