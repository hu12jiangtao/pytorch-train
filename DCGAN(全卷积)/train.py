# 相对于GAN的改进的点:
# 1.判别器使用了LeakyRelu
# 2.为了减小模型的震荡Adam的beta1的参数由0.9该为0.5
# 3.生成器和判别器中都利用了卷积生成网络
# 4.生成器和判别器中增加了BatchNorm(判别器的输入，生成器的输出不需要BatchNorm)
import torch
from torchvision.utils import save_image
import model
import torchvision.transforms as TF
import pathlib
from torch.utils import data
from PIL import Image
from torch import nn

class DataIter(data.Dataset):
    def __init__(self,filenames,transforms):
        super(DataIter, self).__init__()
        self.filenames = filenames
        self.transforms = transforms

    def __getitem__(self, item):
        filename = self.filenames[item]
        image_data = Image.open(filename).convert('RGB')
        if self.transforms:
            image_data = self.transforms(image_data)
        return image_data

    def __len__(self):
        return len(self.filenames)

def load_data(config):
    transforms = TF.Compose([TF.Resize((config.image_size,config.image_size)),
                             TF.ToTensor(),TF.Normalize(mean=[0.5]*3,std=[0.5]*3)])
    path = pathlib.Path(config.input_image)
    filenames = sorted(path.glob('*.jpg'))
    dataset = DataIter(filenames,transforms)
    return data.DataLoader(dataset,batch_size=config.batch_size,shuffle=True,drop_last=True)

def set_grad_require(net,require):
    for param in net.parameters():
        param.requires_grad = require

def train(data_iter, G, D, config):
    loss = nn.BCELoss()
    trainer_G = torch.optim.Adam(G.parameters(),lr=config.lr,betas=(0.5,0.999))
    trainer_D = torch.optim.Adam(D.parameters(), lr=config.lr, betas=(0.5, 0.999))
    real_label = torch.ones(config.batch_size,dtype=torch.float32,device=config.device)
    fake_label = torch.zeros(config.batch_size, dtype=torch.float32, device=config.device)
    batch_iter = 0
    for epoch in range(config.num_epochs):
        for x in data_iter:
            G.train()
            D.train()
            # D的训练
            x = x.to(config.device)
            set_grad_require(D,True)
            noise = torch.randn(size=(config.batch_size,config.channel_noise,1,1),device=config.device)
            fake = G(noise)
            real = x
            fake_out = D(fake.detach()).reshape(-1)
            real_out = D(real).reshape(-1)
            loss_fake = loss(fake_out,fake_label)
            loss_real = loss(real_out,real_label)
            loss_D = (loss_real + loss_fake) / 2
            trainer_D.zero_grad()
            loss_D.backward()
            trainer_D.step()
            # G的训练
            set_grad_require(D, False)
            noise = torch.randn(size=(config.batch_size,config.channel_noise,1,1),device=config.device)
            fake = G(noise)
            out_G = D(fake).reshape(-1)
            loss_G = loss(out_G,real_label)
            trainer_G.zero_grad()
            loss_G.backward()
            trainer_G.step()
            batch_iter += 1
            if batch_iter % 2000 == 0:
                G.eval()
                D.eval()
                print(f'd_loss:{loss_D:.4f}  g_loss:{loss_G:.4f}')
                with torch.no_grad():
                    noise = torch.randn(size=(config.batch_size, config.channel_noise, 1, 1), device=config.device)
                    fake = G(noise)
                    save_image(fake[:10],f'images/{batch_iter}.jpg',nrow=5,normalize=True)



if __name__ == '__main__':
    config = model.Config()
    # 导入数据
    data_iter = load_data(config)
    # 生成器、判别器
    G = model.Generator(config.channel_noise,config.channel_image,config.feature_g).to(config.device)
    D = model.Discriminator(config.channel_image,config.feature_g).to(config.device)
    # 训练
    train(data_iter, G, D, config)


