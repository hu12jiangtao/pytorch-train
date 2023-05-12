import torch
from torch import nn

def init_param(model,method='xavier', exclude='embedding'):
    for name,module in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_uniform_(module)
            elif 'bias' in name:
                nn.init.constant_(module,0)
            else:
                pass

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
    return cmp.type(pred.dtype).sum()

def evaluate_accuracy(data_iter,model):
    model.eval()
    metric = add_machine(3)
    loss = nn.CrossEntropyLoss()
    for x,label in data_iter:
        output = model(x)
        l = loss(output,label)
        metric.add(l * label.numel(), accuracy(output,label), label.numel())
    return metric[0] / metric[2], metric[1] / metric[2]

def train(model,config,train_iter,dev_iter):
    opt = torch.optim.Adam(model.parameters(),lr=config.lr)
    loss = nn.CrossEntropyLoss()
    # 记录参数
    batch_iter = 0
    new_update_batch = 0
    judge_loss = float('inf')
    flag = 0 # 判断训练是否停止的标志位
    # 开始进行训练
    for epoch in range(config.num_epochs):
        print(f'epoch:[{epoch + 1}/{config.num_epochs}]')
        model.train()
        metric = add_machine(3) # 每一次epoch进行重新的计算
        for x,label in train_iter:
            output = model(x)
            l = loss(output,label)
            opt.zero_grad()
            l.backward()
            opt.step()
            metric.add(l * label.numel(),accuracy(output,label), label.numel())
            batch_iter += 1
            if (batch_iter + 1) % 100 == 0:
                train_loss = metric[0] / metric[2]
                train_acc = metric[1] / metric[2]
                dev_loss,dev_acc = evaluate_accuracy(dev_iter,model)
                if dev_loss < judge_loss:
                    judge_loss = dev_loss
                    new_update_batch = batch_iter
                    torch.save(model.state_dict(), config.save_model_dir)
                print(f'batch_iter:{batch_iter + 1} train_loss:{train_loss:1.4f} train_acc:{train_acc:1.4f} dev_loss:{dev_loss:1.4f} dev_acc:{train_acc:1.4f} new_update_batch:{new_update_batch + 1}')
            if batch_iter - new_update_batch > config.judge_batch:
                flag = 1
                break
        if flag == 1:
            break

