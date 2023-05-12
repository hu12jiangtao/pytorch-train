# 整体的大致思路:输入[batch,step]的句子，通过词库将其序列化为[batch,step]的二维矩阵，之后将其embedding为[batch,step,embedding]的三维矩阵
# 对于[batch,step,embedding]输入不同的一维卷积中提取特征后进行拼接得到一个总的特征，这个特征就代表了这句话，之后将特征输入mlp中进行分类

# 目标为对于新闻标题来说将其分为10类(十个不同的子领域)
import argparse
import os
import torch
import util
import importlib
import time
import train_eval

parser = argparse.ArgumentParser()
parser.add_argument('--model',type=str,default='TextCNN')
parser.add_argument('--word',type=bool,default=False,help='true is word, false is char') # 此时的word代表的是词
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained') # embedding利用已经与训练好的权重
args = parser.parse_args()

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True # 每一次的输入都是相同的，因此可以利用该语句进行加速
    # 为预训练的embedding导入进行准备
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model
    dataset = 'THUCNews'
    x = importlib.import_module('model.' + model_name) # 导入model.TextCNN中的所有内容
    # 给定权重参数
    config = x.Config(dataset,embedding) # 找到了model.TextCNN中的Config实例方法，并且将其进行实例化
    # 处理数据集
    start_time = time.time()
    print('loading NEWS...')
    tokens,_ = util.load_file(path=config.train_path, use_word=args.word)
    # 自己创建的字典和其给定的字典的vocab对应关系相同，因此使用调用两个vocab，需要利用不同的方法(flag=0,flag=1)
    vocab = util.create_vocab(config, tokens)  # 读取的字典的长度和config.embedding_pretrained.shape中的参数对应
    # 此时的train_data应当是列表，列表中的元素有这句话的序列、标签、真实长度
    vocab, train_data, dev_data, test_data = util.build_dataset(vocab, config, args.word)
    # 此时的数据集都已经转到gpu上了
    train_iter = util.build_iterator(train_data,config) # 此时的一个元素:(一个batch的序列值构成的二维矩阵,实际长度),label
    dev_iter = util.build_iterator(dev_data,config)
    test_iter = util.build_iterator(test_data,config)
    running_time = util.get_time_dif(start_time)
    print(f"finish loading data, total using {running_time}s")
    # 处理模型
    config.n_vocab = len(vocab)
    use_model = x.Model(config).to(config.device) # model 的输入为(一个batch的序列值构成的二维矩阵,实际长度)
    if model_name != 'transformer':
        train_eval.init_network(use_model)
    use_model.to(config.device)
    # 对模型进行训练
    if not os.path.exists(config.save_path):
        train_eval.train(config, use_model, train_iter, dev_iter, test_iter)
    # 对模型进行测试
    else:
        use_model.load_state_dict(torch.load(config.save_path))  # 导入模型参数
        use_model.eval()
        # 模型输入的格式按照训练时的pad的格式(在test_iter上已经处理好了)
        test_metric = train_eval.add_machine(2)
        for tests,label in test_iter:
            outputs = use_model(tests)
            test_acc = train_eval.accuracy(outputs,label)
            test_metric.add(test_acc,label.numel())
        print(f'Test acc:{test_metric[0] / test_metric[1]:1.4f}')















