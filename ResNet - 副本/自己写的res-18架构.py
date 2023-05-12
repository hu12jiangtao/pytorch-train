import torch
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F
#

d2l.use_svg_display()

#创建一个batch_norm层
def batch_norm(X,gamma,beta,moving_mean,moving_var,momenton=0.9,epsilon=1e-5):
    if not torch.is_grad_enabled():
        X_hat=(X-moving_mean)/(moving_var+epsilon)
    else:
        assert len(X.shape) in (2,4)
        if len(X.shape)==2:
            mean=torch.mean(X,dim=0)
            var=torch.mean((X-mean)**2,dim=0)
        else:
            mean=torch.mean(X,dim=(0,2,3),keepdim=True)
            var=torch.mean((X-mean)**2,dim=(0,2,3),keepdim=True)
        X_hat=(X-mean)/(epsilon+var)
        moving_mean=momenton*moving_mean+(1-momenton)*mean
        moving_var=momenton*moving_var+(1-momenton)*var
    Y=gamma*X_hat+beta
    return Y,moving_mean,moving_var

class BatchNorm(nn.Module):
    def __init__(self,feature_num,num_dims):  # 在卷积层上feature_num=channal_num
        super().__init__()
        if num_dims==2:
            shape=(1,feature_num)
        else:
            shape=(1,feature_num,1,1)
        self.moving_mean=torch.zeros(shape,dtype=torch.float32)
        self.moving_var=torch.zeros(shape,dtype=torch.float32)
        self.gamma=nn.Parameter(torch.ones(shape))
        self.beta=nn.Parameter(torch.zeros(shape))
    def forward(self,X):
        if self.moving_mean.device != X.device:
            self.moving_mean=self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        Y,self.moving_mean,self.moving_var=batch_norm(X, self.gamma, self.beta,
                                                      self.moving_mean, self.moving_var, momenton=0.9, epsilon=1e-5)
        return Y


# 首先创建一个resnet的块，包含着降低figure_size和不降低figure_size两种情况
class resnet_block(nn.Module):
    def __init__(self,in_channal,out_channal,stride,is_change=False):
        super().__init__()
        self.conv1=nn.Conv2d(in_channal,out_channal,stride=stride,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(out_channal,out_channal,stride=1,kernel_size=3,padding=1)
        if is_change:
            self.conv3=nn.Conv2d(in_channal,out_channal,kernel_size=3,padding=1,stride=stride)
        else:
            self.conv3=None
        self.batch_norm1=BatchNorm(out_channal,num_dims=4)
        self.batch_norm2=BatchNorm(out_channal, num_dims=4)
    def forward(self,X):
        Y=F.relu(self.batch_norm1(self.conv1(X)))
        Y=self.batch_norm2(self.conv2(Y))
        if self.conv3:
            X=self.conv3(X)
        Y+=X
        return F.relu(Y)

def resnet_stage(in_channal,out_channal,block_num,is_first=False):
    blk=[]
    for i in range(block_num):
        if i==0 and is_first == False:
            blk.append(resnet_block(in_channal,out_channal,stride=2,is_change=True))
        else:
            blk.append(resnet_block(in_channal, out_channal, stride=1, is_change=False))
        in_channal=out_channal
    return blk

b1=nn.Sequential(nn.Conv2d(1,64,kernel_size=7,padding=3,stride=2),
                 nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
b2=nn.Sequential(*resnet_stage(in_channal=64,out_channal=64,block_num=2,is_first=True))
b3=nn.Sequential(*resnet_stage(in_channal=64,out_channal=128,block_num=2))
b4=nn.Sequential(*resnet_stage(in_channal=128,out_channal=256,block_num=2))
b5=nn.Sequential(*resnet_stage(in_channal=256,out_channal=512,block_num=2))

net=nn.Sequential(b1,b2,b3,b4,b5,nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),nn.Linear(512,10))


X=torch.randn((1,1,224,224))
for layer in net:
    X=layer(X)
    print(layer.__class__.__name__,'\toutput shape:',X.shape)