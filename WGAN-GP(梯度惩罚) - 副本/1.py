import torch
from torch import nn

class mode(nn.Module):
    def __init__(self):
        super(mode, self).__init__()
        self.linear = nn.Linear(4,3)
    def forward(self,x):
        return self.linear(x)

net = mode()
for p in net.parameters():
    print(p.requires_grad)

alpha = torch.rand((64, 1, 1, 1)).repeat(1, 1, 32, 32)
print(alpha[0] == alpha[1])

# 对于autograd的测试
a = torch.tensor(10.,requires_grad=True)
b = torch.tensor(2.,requires_grad=True)
c = a * b
grad = torch.autograd.grad(inputs=a,outputs=c,grad_outputs=torch.ones_like(b))
print(grad)


def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    # interpolated_images代表着在X_penalty中采样的点
    interpolated_images = real * alpha + fake * (1 - alpha) #由于其中的一个参数fake是requires_grad=True的因此interpolated_images是True的

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)
    # Take the gradient of the scores with respect to the images
    # torch.autograd.grad是针对于张量对张量的求导，grad_outputs应当和outputs形状相同，用来分配每一项在求导时占据的权重，其中的0代表着权重参数的信息存放在[0]中
    # gradient中只存放着inputs中每个像素对应的梯度
    # 虽然此时的D的参数和输入都是叶子节点(requires_grad=True,但是都是用户自己设计)，
    # 但是torch.autograd.grad是只将inputs作为未知参数，D的网络参数作为已知量进行求导，因此输出应当和输入的形状相同
    gradient = torch.autograd.grad(
        inputs=interpolated_images,  # inputs必须requires_grad=True
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]  # 得到的结果[batch,c,H,W]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty



alpha = torch.rand((3, 1, 1, 1)).repeat(1, 1,4,4)
print(alpha)