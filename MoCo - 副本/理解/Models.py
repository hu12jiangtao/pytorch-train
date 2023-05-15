import torch
from torch import nn
from torch.nn import functional as F

# Models中存放的是resnet函数(作为编码器)
class BasicBlock(nn.Module):
    def __init__(self,in_channel,out_channel,stride,dropout):
        super(BasicBlock, self).__init__()
        self.norm1 = nn.BatchNorm2d(in_channel)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=stride,padding=1,bias=False)

        self.norm2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel,out_channel,kernel_size=3,stride=1,padding=1,bias=False)

        self.dropout = dropout
        self.InOutEqual = (in_channel == out_channel)
        self.shortCut = (not self.InOutEqual and nn.Conv2d(in_channel,out_channel,
                                                           kernel_size=1,stride=stride,padding=0,bias=False)) or None

    def forward(self,x):
        if self.InOutEqual:
            out = self.relu1(self.norm1(x))
        else:
            x = self.relu1(self.norm1(x))
        out = self.relu2(self.norm2(self.conv1(out if self.InOutEqual else x)))
        out = F.dropout(out,self.dropout,training=self.training)
        out = self.conv2(out)
        return torch.add(out, x if self.InOutEqual else self.shortCut(x))

class NetworkBlock(nn.Module): # 每个网络模块只有第一个模块是会改变图片宽高和通道数量的
    def __init__(self,num_layers,in_channel,out_channel,block,stride,dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.net = self._make_layer(num_layers,in_channel,out_channel,block, stride, dropRate)

    def _make_layer(self, num_layers,in_channel,out_channel,block, stride, dropRate):
        layers = []
        for i in range(num_layers):
            layers.append(block((i == 0 and in_channel) or out_channel,out_channel,(i == 0 and stride) or 1,dropRate))
        return nn.Sequential(*layers)

    def forward(self,x):
        return self.net(x)

class WideResNet(nn.Module):
    def __init__(self, pretrained=None, num_classes=None, depth=28, widen_factor=2, dropRate=0.0):
        super(WideResNet, self).__init__()
        # 作为一个encode应当满足输入一张图片，给出一个对应的特征向量
        num_layers = int((depth - 4) / 6)
        n_channels = [16, widen_factor * 16, widen_factor * 32, widen_factor * 64]
        self.conv1 = nn.Conv2d(3, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        block = BasicBlock
        self.block1 = NetworkBlock(num_layers, n_channels[0], n_channels[1], block, 1, dropRate)
        self.block2 = NetworkBlock(num_layers, n_channels[1], n_channels[2], block, 2, dropRate)
        self.block3 = NetworkBlock(num_layers, n_channels[2], n_channels[3], block, 2, dropRate)
        self.bn1 = nn.BatchNorm2d(n_channels[-1])
        self.relu = nn.ReLU(inplace=True)
        self.mlp = nn.Sequential(nn.Linear(n_channels[-1],n_channels[-1]),
                                 nn.ReLU(inplace=True),nn.Linear(n_channels[-1],n_channels[-1]))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def features(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        return out

    def forward(self,x):
        out = self.features(x)
        out = F.adaptive_avg_pool2d(out,(1,1))
        out = out.reshape(out.shape[0], -1)
        out = self.mlp(out)
        # 由于利用MoCo无监督学习的特征和利用有监督学习的预训练模型得到的特征差异较大(无监督训练的特征在微调的时候的初始学习率为30)，因此需要进行归一化操作
        # F.normalize(out)相当于按照行进行归一化操作，此时利用无监督学习训练到的特征在下游任务上进行微调的学习率和有监督的相近似，而不是初始参数这么大的情况
        return F.normalize(out) # [batch, 128]



if __name__ == '__main__':
    torch.manual_seed(1)
    x = torch.randn(size=(1,3,32,32))
    net = WideResNet(pretrained=None)
    print(net(x))