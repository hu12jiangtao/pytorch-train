import os
import torch
from torch import nn
import argparse
import importlib
import util
import time
import train_eval

phase = argparse.ArgumentParser()
phase.add_argument('--model',type=str,default='TextCNN')
phase.add_argument('--use_word',type=bool,default=False)
phase.add_argument('--embedding',type=str,default='pre_trained',choices=['pre_trained','random'])
args = phase.parse_args()

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True  # 利用这条语句输入的形状应当完全相同(batch野应当是相同的)
    dataset = 'D:\\python\\pytorch作业\\nlp实战练习\\Text_CNN\\textcnn\\THUCNews'
    model_file = 'model.' + args.model
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'
    x = importlib.import_module(model_file)
    path = os.getcwd()
    config = x.Config(path,dataset,embedding)
    # 创建数据集
    start_time = time.time()
    tokens,labels = util.load_file(config.train_dir)
    tokens = util.tokenize(tokens,args.use_word)
    vocab = util.choice_vocab(tokens,config)
    train_data, dev_data, test_data = util.build_datsaet(config, vocab, use_word=args.use_word)
    train_iter = util.build_iterator(train_data, config)
    dev_iter = util.build_iterator(dev_data,config)
    test_iter = util.build_iterator(test_data,config)
    config.vocab_size = len(vocab)
    print(f'load dataset using : {util.spend_time(start_time)}s')
    # 创建模型
    model = x.Model(config)
    train_eval.init_param(model)
    model.to(config.device)
    # 训练和测试
    if not os.path.exists(config.save_model_dir):
        train_eval.train(model, config, train_iter, dev_iter)
    else:
        model.load_state_dict(torch.load(config.save_model_dir))
    with torch.no_grad():
        model.eval()
        test_metric = train_eval.add_machine(2)
        for x,label in test_iter:
            output = model(x)
            test_metric.add(train_eval.accuracy(output,label),label.numel())
        print(f'test_acc:{test_metric[0] / test_metric[1]:1.4f}')











