import os
import importlib
import collections
import pickle
import time

import torch


def load_file(path):
    with open(path,'r',encoding='utf-8') as f:
        file = f.readlines()
    data = [line.strip().split('\t')[0] for line in file]
    labels = [line.strip().split('\t')[1] for line in file]
    return data,labels

def tokenize(data,mode):
    if not mode:
        tokens = [list(line) for line in data]
    else:
        raise NotImplementedError('this solution only support char')
    return tokens

def collect_num(tokens):
    if isinstance(tokens[0],(tuple,list)):
        tokens = [token for line in tokens for token in line]
    else:
        tokens = [token for token in tokens]
    return collections.Counter(tokens)

class Vocab(object):
    def __init__(self,tokens,max_size,min_freq):
        if tokens is None:
            tokens = []
        collect = collect_num(tokens)
        self.token_collect = sorted(collect.items(),key=lambda x:x[1],reverse=True)
        self.token_lst = []
        self.token_lst += [name for name,num in self.token_collect if name not in self.token_lst and num >= min_freq]
        self.token_lst = self.token_lst[:max_size]
        self.tokens_to_idx,self.idx_to_tokens = {},[]
        for name in self.token_lst:
            self.idx_to_tokens.append(name)
            self.tokens_to_idx[name] = len(self.idx_to_tokens) - 1
        self.tokens_to_idx['<UNK>'] = len(self.idx_to_tokens)
        self.tokens_to_idx['<PAD>'] = len(self.idx_to_tokens) + 1
        self.idx_to_tokens += ['<UNK>','<PAD>']

    def __getitem__(self, item):
        if not isinstance(item,(list,tuple)):
            return self.tokens_to_idx.get(item,self.tokens_to_idx['<UNK>'])
        else:
            return [self.__getitem__(i) for i in item]

    def to_tokens(self,item):
        if not isinstance(item,(list,tuple)):
            return self.idx_to_tokens[item]
        else:
            return [self.idx_to_tokens[i] for i in item]

    def __len__(self):
        return len(self.idx_to_tokens)

def choice_vocab(vocab_path,tokens):
    if not os.path.exists(vocab_path):
        vocab = Vocab(tokens,max_size=10000,min_freq=1)
        pickle.dump(vocab,open(vocab_path,'wb'))
    else:
        vocab = pickle.load(open(vocab_path,'rb'))
    print(f"Vocab size: {len(vocab)}")
    return vocab

def deal_dataset(tokens,labels,vocab,config): # 需要对seq长度不同的每一个样本做批量的处理(对于训练集、验证机、测试集都需要进行)
    save = []
    tokens = tokenize(tokens,mode=config.use_word)
    for i in range(len(tokens)):
        token = tokens[i]
        label = labels[i]
        fact_seq = len(token)
        if fact_seq < config.pad_size:
            token = token + ['<PAD>'] * (config.pad_size - fact_seq)
        else:
            token = token[:config.pad_size]
            fact_seq = config.pad_size
        token = vocab[token]
        save.append((token,int(label),fact_seq))
    return save

def create_dataset(vocab,config):
    train_tokens,train_labels = load_file(config.train_path)
    train = deal_dataset(train_tokens,train_labels,vocab, config)

    dev_tokens,dev_labels = load_file(config.dev_path)
    dev = deal_dataset(dev_tokens,dev_labels,vocab,config)

    test_tokens,test_labels = load_file(config.test_path)
    test = deal_dataset(test_tokens,test_labels,vocab,config)
    return train,dev,test

class Dataset_iter(object): # 抛弃剩余部分
    def __init__(self,batches,batch_size,device):
        self.batches = batches
        self.batch_size = batch_size
        self.n_batch = len(self.batches) // self.batch_size
        self.device = device
        self.index = 0

    def _to_tensor(self,data):
        input_data = torch.tensor([_[0] for _ in data],dtype=torch.long,device=self.device)
        labels = torch.tensor([_[1] for _ in data],dtype=torch.long,device=self.device)
        fact_seq = torch.tensor([_[2] for _ in data],dtype=torch.long,device=self.device)
        return (input_data,fact_seq),labels

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.n_batch:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size:(self.index + 1) * self.batch_size]
            batches = self._to_tensor(batches)
            self.index += 1
            return batches
    def __len__(self):
        return len(self.n_batch)

def create_iter(batches,config):
    iter = Dataset_iter(batches,config.batch_size,config.device)
    return iter

def spend_time(start_time):
    end_time = time.time()
    use_time = end_time - start_time
    return round(use_time)

if __name__ == '__main__':
    work_path = os.getcwd()
    x = importlib.import_module('model.TextRNN')
    data_dir = 'D:\\python\\pytorch作业\\nlp实战练习\\Text_CNN\\textcnn\\THUCNews'
    embedding = 'pre_train'
    config = x.Config(data_dir,embedding,work_path)
    data,labels = load_file(config.train_path)
    tokens = tokenize(data,False)
    vocab = choice_vocab(config.vocab_path, tokens)
