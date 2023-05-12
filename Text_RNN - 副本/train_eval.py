import torch
from torch import nn

def init_param(model,mode='xavier',excluding='embedding'):
    for name,module in model.named_parameters():
        if excluding not in name:
            if 'weight' in name:
                if mode == 'xavier':
                    nn.init.xavier_uniform_(module)
            elif 'bias' in name:
                nn.init.constant_(module,0)
            else:
                pass
        else:
            continue

class add_machine():
    def __init__(self,n):
        self.data = [0.] * n
    def add(self,*args):
        self.data = [i + float(j) for i,j in zip(self.data,args)]
    def __getitem__(self, item):
        return self.data[item]

def accuracy(y_hat,y):
    pred = torch.argmax(y_hat,dim=-1)
    cmp = (pred.type(y.dtype) == y)
    return cmp.type(y.dtype).sum()

def evaluate_accuracy(config,data_iter,model):
    # 训练过程中只训练LSTM的参数，state不管实在验证还是训练的情况下都是从头还是随机赋值
    model.eval()
    loss = nn.CrossEntropyLoss()
    metric = add_machine(3)
    for dev_x, dev_y in data_iter:
        y_hat,state = model(dev_x)
        l = loss(y_hat, dev_y)
        metric.add(l * dev_y.numel(), accuracy(y_hat,dev_y), dev_y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


def train(train_iter,dev_iter,model,config):
    # 训练方式
    opt = torch.optim.Adam(model.parameters(),lr=config.lr)
    loss = nn.CrossEntropyLoss()
    # 记录参数设置
    batch_iter = 0
    update_loss = float('inf')
    update_iter = 0
    flag = 0
    # 开始训练
    for epoch in range(config.sum_epoch):
        print(f'Epoch [{epoch + 1}/{config.sum_epoch}]')
        metric = add_machine(3)
        model.train()
        for x,y in train_iter:
            y_hat, state = model(x)
            l = loss(y_hat,y)
            opt.zero_grad()
            l.backward()
            opt.step()
            metric.add(l * y.numel(), accuracy(y_hat,y), y.numel())
            batch_iter += 1
            if batch_iter % 100 == 0:
                train_loss = metric[0] / metric[2]
                train_acc = metric[1] / metric[2]
                dev_loss,dev_acc = evaluate_accuracy(config,dev_iter,model)
                if dev_loss < update_loss:
                    update_loss = dev_loss
                    update_iter = batch_iter
                    torch.save(model.state_dict(),config.save_model_path)
                print(f'Iter:{batch_iter} Train_Loss:{train_loss:1.4f} Train_acc:{train_acc:1.4f} Dev_Loss:{dev_loss:1.4f} Dev_acc:{dev_acc:1.4f},update_iter:{update_iter}')
                model.train()
            if batch_iter - update_iter > config.requirement:
                flag = 1
                break
        if flag == 1:
            break







