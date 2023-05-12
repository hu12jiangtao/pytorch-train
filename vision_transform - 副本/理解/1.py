from torch import nn
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(4,3)
        self.linear2 = nn.Linear(3,2)
    def forward(self,x):
        return self.linear2(self.linear1(x))

device = torch.device('cuda')
net = Net().to(device)
loss = nn.CrossEntropyLoss()
opt = torch.optim.Adam(net.parameters(),lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,10)
x = torch.randn(size=(5,4),device=device)
y = torch.tensor([1,0,0,1,1],device=device)
for i in range(20):
    x, y = x.to(device), y.to(device)
    y_hat = net(x)
    l = loss(y_hat,y)
    opt.zero_grad()
    l.backward()
    opt.step()
    scheduler.step()
    print(opt.param_groups[0]['lr'])