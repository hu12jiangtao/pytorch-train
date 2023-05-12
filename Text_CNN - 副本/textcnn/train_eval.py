from torch import nn
import torch
import model.TextCNN as model

def init_network(model,method='xavier', exclude='embedding'):  # 对于embedding层来说不进行初始化
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
    return cmp.type(y.dtype).sum()

def accuracy_evaluate(config, model, dev_iter):
    model.to(config.device)
    model.eval()
    loss_total = 0
    with torch.no_grad():
        metric = add_machine(3)
        for texts,label in dev_iter:
            outputs = model(texts)
            loss = nn.CrossEntropyLoss()
            l = loss(outputs,label)
            loss_total += l
            metric.add(accuracy(y_hat=outputs,y=label),loss_total,label.numel())
    return metric[0]/metric[2], metric[1] / metric[2]  # 获得整个数据集的准确率和平均损失


def train(config, use_model, train_iter, dev_iter, test_iter):
    use_model.train()
    opt = torch.optim.Adam(use_model.parameters(),lr=config.learning_rate)
    loss = nn.CrossEntropyLoss()\

    total_batch = 0
    dev_best_loss = float('inf')
    last_improve = 0  # 用来记录最新一次导入模型batch数量
    flag = False

    for epoch in range(config.num_epochs):
        print(f'Epoch [{epoch + 1}/{config.num_epochs}]')
        metric = add_machine(3)
        use_model.train()
        for i,(trains, label) in enumerate(train_iter):
            outputs = use_model(trains)
            l = loss(outputs,label)
            opt.zero_grad()
            l.backward()
            opt.step()
            metric.add(l * label.numel(), accuracy(outputs,label), label.numel())
            total_batch += 1
            if total_batch % 100 == 0:
                # 对于训练集和验证集的检测
                train_acc = metric[1] / metric[2] # 训练集的准确率
                dev_acc,dev_loss = accuracy_evaluate(config,use_model,dev_iter) # 测试集的准确率和损失
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(use_model.state_dict(),config.save_path) # 存储模型
                    last_improve = total_batch
                print(f'Iter:{total_batch} Train_Loss:{metric[0] / metric[2]:1.4f} Train_acc:{train_acc:1.4f} Dev_Loss:{dev_loss:1.4f} Dev_acc:{dev_acc:1.4f},last_improve:{last_improve}')
            if total_batch - last_improve >= config.require_improvement:
                flag = True
                break
        if flag:
            break
        print('*'*100)










if __name__ == '__main__':
    config = model.Config(dataset = 'THUCNews',embedding='embedding_SougouNews.npz')
    m = model.Model(config)
    init_network(m)