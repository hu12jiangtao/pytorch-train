# 在此训练过程中，token的取值方式为按字取值
import os
import collections
import pickle
import time
import torch

def load_file(path):
    with open(path,'r',encoding='utf-8') as f:
        file = f.readlines()
    texts = [line.strip().split('\t')[0] for line in file]
    labels = [line.strip().split('\t')[1] for line in file]
    return texts,labels

def tokenize(texts,use_word):
    if not use_word:
        texts = [list(line) for line in texts]
    else:
        raise NotImplementedError('must use char ,word is not define !')
    return texts

def collect_num(tokens):
    if isinstance(tokens[0],list):
        tokens = [token for line in tokens for token in line]
    else:
        tokens = [token for token in tokens]
    return collections.Counter(tokens)

class Vocab(object): # 将'<UNK>','<PAD>'放置在最后面
    def __init__(self,tokens,max_size=None,limit_seq=None):
        if tokens is None:
            tokens = []
        collect = collect_num(tokens)
        self.token_collect = sorted(collect.items(),key=lambda x:x[1],reverse=True)
        self.tokens_lst = []
        self.tokens_lst += [name for name,num in self.token_collect if name not in self.tokens_lst and num >= limit_seq]
        self.tokens_lst = self.tokens_lst[:max_size]
        self.tokens_to_idx,self.idx_to_tokens = {},[]
        for name in self.tokens_lst:
            self.idx_to_tokens.append(name)
            self.tokens_to_idx[name] = len(self.idx_to_tokens) - 1
        self.idx_to_tokens.extend(['<UNK>','<PAD>'])
        self.tokens_to_idx['<UNK>'] = len(self.idx_to_tokens)
        self.tokens_to_idx['<PAD>'] = len(self.idx_to_tokens) + 1
    def __getitem__(self, item):
        if not isinstance(item,(tuple,list)):
            return self.tokens_to_idx.get(item,self.tokens_to_idx['<UNK>'])
        else:
            return [self.__getitem__(i) for i in item]
    def to_tokens(self,item):
        if not isinstance(item, (tuple, list)):
            return self.idx_to_tokens[item]
        else:
            return [self.idx_to_tokens[i] for i in item]
    def __len__(self):
        return len(self.idx_to_tokens)

def choice_vocab(tokens,config):
    if os.path.exists(config.vocab_dir):
        vocab = pickle.load(open(config.vocab_dir,'rb'))
    else:
        vocab = Vocab(tokens,config.max_size,config.limit_seq)
        with open(config.vocab_dir,'w') as f:
            pickle.dump(vocab,f)
    print(f'vocab size:{len(vocab)}')
    return vocab

# 使得每个batch的seq长度相同
def load_dataset(tokens,labels,vocab,pad_size):
    contents = []
    for i in range(len(labels)):
        token = tokens[i]
        label = labels[i]
        fact_seq_len = len(token)
        if fact_seq_len < pad_size:
            token = token + ['<PAD>'] * (pad_size - fact_seq_len)
        else:
            token = token[:pad_size]
            fact_seq_len = pad_size
        token = vocab[token] # 进行序列化
        contents.append((token,int(label),fact_seq_len))
    return contents

# 用于创建数据集
def build_datsaet(config,vocab,use_word):
    train_texts,train_label = load_file(config.train_dir)
    train_tokens = tokenize(train_texts,use_word)
    train = load_dataset(train_tokens, train_label, vocab, config.pad_size)

    dev_texts,dev_label = load_file(config.dev_dir)
    dev_tokens = tokenize(dev_texts,use_word)
    dev = load_dataset(dev_tokens, dev_label, vocab, config.pad_size)

    test_texts,test_label = load_file(config.test_dir)
    test_tokens = tokenize(test_texts,use_word)
    test = load_dataset(test_tokens, test_label, vocab, config.pad_size)
    return train,dev,test

class DatasetIterater(object):
    def __init__(self,batches,batch_size,device): # batches为一个列表，其中元素为()
        self.batches = batches
        self.batch_size = batch_size
        self.device = device
        self.n_batch = len(batches) // batch_size
        self.index = 0 # 记录当前是哪一个batch的
        if len(batches) // batch_size != 0:
            self.flag = 1
        else:
            self.flag = 0

    def to_tensor(self,data):
        x = torch.tensor([_[0] for _ in data],dtype=torch.long,device=self.device)
        y = torch.tensor([_[1] for _ in data],dtype=torch.long,device=self.device)
        fact_seq_len = torch.tensor([_[2] for _ in data],dtype=torch.long,device=self.device)
        return (x,fact_seq_len),y

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.n_batch:  # 对于len(batches) % batch_size剩余的部分应当做舍弃处理
            self.index = 0
            raise StopIteration
        else:
            date = self.batches[self.index * self.batch_size:(self.index + 1) * self.batch_size]
            data = self.to_tensor(date)
            self.index += 1
            return data
    def __len__(self):
        if self.flag == 0:
            return self.n_batch
        else:
            return self.n_batch + 1

def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter

def spend_time(start_time):
    end_time = time.time()
    use_time = end_time - start_time
    return round(use_time)


if __name__ == '__main__':
    dateset = dataset = 'D:\\python\\pytorch作业\\nlp实战练习\\Text_CNN\\textcnn\\THUCNews'
    path = os.path.join(dataset, 'NEWS/train.txt')
    texts,labels = load_file(path)
    tokens = tokenize(texts,False)
