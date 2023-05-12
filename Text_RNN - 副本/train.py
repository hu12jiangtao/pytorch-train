# 双向LSTM不适用的情况:生成模型(只能通过前面内容来预测后面的内容)，对于翻译的编码部分、情感分析部分应当为双向LSTM优于单项的LSTM

# 对于双向LSTM来说:input = [batch,seq,embed_size]->permute(1,0,2):[seq,batch,embed_size]
# 初始的隐藏层: (隐藏单元[num_layer * num_direction, batch, num_hidden],记忆单元[num_layer * num_direction, batch, num_hidden])

# 输出(没有经过全连接): output = [seq,batch,num_hidden * num_direction]
import time
import torch
import argparse
import importlib
import os
import train_eval
import util

phase = argparse.ArgumentParser()
phase.add_argument('--use_word',type=bool,default=False)
phase.add_argument('--model',type=str,default='TextRNN')
phase.add_argument('--pretrain_embed',type=str,choices=['random','embedding_SougouNews.npz'],default='embedding_SougouNews.npz')
args = phase.parse_args()

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    model_path = 'model.' + args.model
    x = importlib.import_module(model_path)
    data_dir = 'D:\\python\\pytorch作业\\nlp实战练习\\Text_CNN\\textcnn\\THUCNews'
    work_path = os.getcwd()
    embedding = args.pretrain_embed
    config = x.Config(data_dir,embedding,work_path)
    config.use_word = args.use_word
    # 加载数据集
    start_time = time.time()
    data,labels = util.load_file(config.train_path)
    tokens = util.tokenize(data,False)
    vocab = util.choice_vocab(config.vocab_path, tokens)
    train,dev,test = util.create_dataset(vocab, config) # 三个数据集
    train_iter = util.create_iter(train,config)
    dev_iter = util.create_iter(dev,config)
    test_iter = util.create_iter(test,config)
    config.vocab_size = len(vocab)
    use_time = util.spend_time(start_time)
    print(f'load dataset using {use_time}s')
    # 创建模型
    model = x.Model(config)
    if args.model != 'transformer':
        train_eval.init_param(model)
    model.to(config.device)
    if os.path.exists(config.save_model_path):
        model.load_state_dict(torch.load(config.save_model_path))
    else:
        train_eval.train(train_iter,dev_iter,model,config)
    # 开始测试
    model.eval()
    test_metric = train_eval.add_machine(2)
    for x,y in test_iter:
        y_hat,_ = model(x)
        test_metric.add(train_eval.accuracy(y_hat,y),y.numel())
    print(f'test acc:{test_metric[0] / test_metric[1]:1.4f}')




