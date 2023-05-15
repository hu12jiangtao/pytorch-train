from torch import nn
import torch

class Config(object):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_image = 'faces'
        self.image_size = 128
        self.batch_size = 20
        self.channel_noise = 100
        self.channel_image = 3
        self.feature_g = 16
        self.lr = 2e-4
        self.num_epochs = 10000

class Generator(nn.Module): # 利用反卷积
    def __init__(self,channel_noise, channel_image, feature_g):
        super(Generator, self).__init__()
        ''' input: [batch,channel_noise,1,1] '''
        self.module = nn.Sequential(
            self.blk(channel_noise,feature_g * 16,4,1,0), # [batch,feature_g * 16,4,4]
            self.blk(feature_g * 16, feature_g * 8, 4, 2, 1),  # [batch,feature_g * 8,8,8]
            self.blk(feature_g * 8, feature_g * 4, 4, 2, 1),  # [batch,feature_g * 4,16,16]
            self.blk(feature_g * 4, feature_g * 2, 4, 2, 1),  # [batch,feature_g * 2,32,32]
            self.blk(feature_g * 2, feature_g, 4, 2, 1),  # [batch,feature_g,64,64]
            nn.ConvTranspose2d(feature_g,channel_image,kernel_size=4,stride=2,padding=1), # [batch,3,128,128]
            nn.Tanh()
        )

    def blk(self,in_channel,out_channel,k,s,p,bias=False):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channel,out_channel,kernel_size=k,stride=s,padding=p,bias=bias),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self,x):
        return self.module(x)

class Discriminator(nn.Module):
    def __init__(self, channel_image, feature_g):
        super(Discriminator, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(channel_image,feature_g,kernel_size=4,stride=2,padding=1), # [batch,feature_g,64,64]
            nn.LeakyReLU(0.2),
            self.blk(feature_g, feature_g * 2, 4, 2, 1), # [batch,feature_g * 2,32,32]
            self.blk(feature_g * 2, feature_g * 4, 4, 2, 1), # [batch,feature_g * 4,16,16]
            self.blk(feature_g * 4, feature_g * 8, 4, 2, 1), # [batch,feature_g * 8,8,8]
            self.blk(feature_g * 8, feature_g * 16, 4, 2, 1),  # [batch,feature_g * 16,4,4]
            nn.Conv2d(feature_g * 16, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def blk(self,in_channel,out_channel,k,s,p,bias=False):
        return nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=k,stride=s,padding=p,bias=bias),
            nn.BatchNorm2d(out_channel,affine=True),
            nn.LeakyReLU(0.2)
        )

    def forward(self,x):
        return self.module(x)

def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
