import torch
from torch.autograd import Variable
a = torch.tensor(2.0,requires_grad = True)
b = torch.tensor(3.0)
c = a*b
c.backward()
print(a.grad)