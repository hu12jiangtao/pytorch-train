import torchvision
from torch import nn
import torch

# pretrain_net = torchvision.models.resnet18(pretrained=True)
# # print(list(net.children())[:-1])
# x = torch.randn(size=(1,3,32,32))
# net = nn.Sequential()
# net.bone = nn.Sequential(*list(pretrain_net.children())[:-1])
# out = net.bone(x)
# print(out.shape)
# out = out.reshape(out.shape[0],-1)
# print(out.shape)
# net.fc = nn.Linear(out.shape[-1], 10)
# print(net.fc(out).shape)


# net_2 = list(net.children())[:-1]
# print(nn.Sequential(*net_2)(x).shape)

net1 = nn.Linear(4,3)
opt = torch.optim.SGD(net1.parameters(),lr=0.01)
state = {'net1':net1.state_dict(), 'opt':opt.state_dict()}
torch.save(state,'1.params')

net1.load_state_dict(torch.load('1.params')['net1'])
