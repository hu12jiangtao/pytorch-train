# Incception网络中3*3的卷积，或者5*5的卷积前面的1*1的卷积块的作用都是用来降低通道数的，通过机器来选择这一层时使用的是3*3卷积还是5*5卷积还是1*1卷积还是maxpool

# 选用1*1卷积块的另一个用处就是减少计算力，例如输入为192*28*28，将其转换为128*28*28（直接利用3*3大小的核，存在padding），直接计算的乘法数量为128*28*28*3*3*192
# 加入一个1*1的卷积（96个通道）后计算量为1*1*192*96*28*28+3*3*96*28*28*128

# Inception模块的输出通道数有多个通道组合形成的，一般3*3的卷积提供的通道数较多，计算代价不大，也能抽取较好的空间信息，
# 而连接3*3的卷积层的1*1卷积层的通道数一般是3*3的卷积提供的通道数的一般或者是0.75左右

# goole net 和 Nin net 的最后都是一样的，并没有使用全连接层，而是利用了一个全局平均池化层来将figture_size降到(1,1)
# goole net 的stage1中使用的卷积为7*7的其他的都是3*3，1*1的卷积，或者是inception块

# 通道数的选择最好选的是2^n次方，这样子GPU计算的较为方便

from torch import nn
import torch
from torch.nn import functional as F

def add(*args,**kwargs):  # *args是指不知道输入的参数的个数，因此将所有输入的参数转换为元组输入到函数中
    print(args)
    print(kwargs)   # **kwargs是指不知道输入的参数的个数，以关键字赋值的形式赋值给参数，输入到函数中是以字典的形式
add(10,12,name='Alice',age=17,gender=2)
print('----------------------------------------------------')
class Person():
    def __init__(self,name,weight,age):
        self.name=name
        self.weight=weight
        self.age=age

class student(Person):
    def __init__(self,gender,school,**kwargs):
        super(student, self).__init__(**kwargs) # 此时是对父类初始化的内容的继承，由于不知道继承多少，因此利用**kwargs的字典形式导入
        self.gendef=gender
        self.school=school
        print(kwargs)

s = student(18,'shabi',name='Alice',weight=87,age=17)

print('----------------------------------------------------')

class Inception(nn.Module):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        self.p1_1=nn.Conv2d(in_channels,c1,kernel_size=1)
        self.p2_1=nn.Conv2d(in_channels,c2[0],kernel_size=1)
        self.p2_2=nn.Conv2d(c2[0],c2[1],kernel_size=3,padding=1)
        self.p3_1=nn.Conv2d(in_channels,c3[0],kernel_size=1)
        self.p3_2=nn.Conv2d(c3[0],c3[1],kernel_size=5,padding=2)
        self.p4_1=nn.MaxPool2d(kernel_size=3,padding=1,stride=1)
        self.p4_2=nn.Conv2d(in_channels,c4,kernel_size=1)
    def forward(self,X):
        out1=F.relu(self.p1_1(X))
        print(out1.shape)
        out2=F.relu(self.p2_2(F.relu(self.p2_1(X))))
        print(out2.shape)
        out3=F.relu(self.p3_2(F.relu(self.p3_1(X))))
        print(out3.shape)
        out4=F.relu(self.p4_2(self.p4_1(X)))
        print(out4.shape)
        out=torch.cat((out1,out2,out3,out4),1)
        return out


X=torch.randn((1,192,28,28),dtype=torch.float32)
inception=Inception(192,64,(96,128),(16,32),32)
net=nn.Sequential(inception)
print(net(X))
