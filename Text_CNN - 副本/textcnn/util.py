import os
import pickle
import torch
import collections
import time

def load_file(path,use_word): # 此时使用的是按字进行分类
    with open(path,'r',encoding='utf-8') as f:
        file_data = f.readlines()
    tokens = [line.strip().split('\t')[0] for line in file_data if line]
    labels = [line.strip().split('\t')[1] for line in file_data if line]
    if not use_word:
        tokens = [list(line) for line in tokens]
    else:
        raise NotImplementedError('must input char type')
    return tokens,labels

def collextion_num(tokens):
    if isinstance(tokens[0],list):
        tokens = [token for line in tokens for token in line]
    else:
        tokens = [token for token in tokens]
    return collections.Counter(tokens)

class Vocab():
    def __init__(self,tokens,max_size,min_freq):
        if tokens is None:
            tokens = []
        self.collect = collextion_num(tokens)
        self.seq_collect = sorted(self.collect.items(),key=lambda x:x[1],reverse=True)
        self.token_lst = []
        self.token_lst += [name for name,nums in self.seq_collect if nums >= min_freq and name not in self.token_lst]
        self.token_lst = self.token_lst[:max_size]
        self.idx_to_tokens,self.tokens_to_idx = [],{}
        for name in self.token_lst:
            self.idx_to_tokens.append(name)
            self.tokens_to_idx[name] = len(self.idx_to_tokens) - 1
        self.tokens_to_idx['<UNK>'] = len(self.idx_to_tokens)
        self.tokens_to_idx['<PAD>'] = len(self.idx_to_tokens) + 1
        self.idx_to_tokens += ['<UNK>','<PAD>']

    def __getitem__(self, item):
        if not isinstance(item,(tuple,list)):
            return self.tokens_to_idx.get(item,self.tokens_to_idx['<UNK>'])
        else:
            return [self.__getitem__(i) for i in item]

    def to_tokens(self,index):
        if not isinstance(index,(tuple,list)):
            return self.idx_to_tokens[index]
        else:
            return [self.idx_to_tokens[i] for i in index]

    def __len__(self):
        return len(self.idx_to_tokens)

def load_dataset(tokens,labels,vocab,pad_size=32):
    contents = []
    for i in range(len(tokens)):
        token = tokens[i]
        label = labels[i]
        seq_len = len(token)
        if pad_size:
            if len(token) < pad_size:
                token += ['<PAD>'] * (pad_size - len(token))
            else:
                token = token[:pad_size]
                seq_len = pad_size
            # 将token转换为序列
            save_token = vocab[token]
            contents.append((save_token,int(label),seq_len))
    return contents

def build_dataset(vocab, config, use_word):
    train_tokens, train_labels = load_file(path=config.train_path, use_word=use_word)
    train = load_dataset(train_tokens,train_labels,vocab,config.pad_size)

    dev_tokens, dev_labels = load_file(path=config.dev_path, use_word=use_word)
    dev = load_dataset(dev_tokens,dev_labels,vocab,config.pad_size)

    test_tokens, test_labels = load_file(path=config.test_path, use_word=use_word)
    test = load_dataset(test_tokens,test_labels,vocab,config.pad_size)

    return vocab,train,dev,test

def create_vocab(config,tokens): # ues_word代表模式，T代表word，F代表char
    if os.path.exists(config.vocab_path):
        vocab = pickle.load(open(config.vocab_path,'rb'))
    else:
        vocab = Vocab(tokens,max_size=10000, min_freq=1)
        pickle.dump(vocab,open(config.vocab_path,'wb'))
    print(f"Vocab size: {len(vocab)}")
    return vocab

class DatasetIterater(object):
    def __init__(self,batches,batch_size,device):
        self.batch_size = batch_size
        self.batches = batches
        self.device = device
        self.n_batches = len(batches) // batch_size  # 一共存在多少个batch
        self.residue = False  # 所有的数据集是否能被整除
        if len(batches) % batch_size != 0:
            self.residue = False
        self.index = 0

    def _to_tensor(self,datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device) # 此时取出了序列部分作为标签 [batch,pad_size]
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device) # 此时代表着标签[batch,]
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device) # 每个batch的实际长度
        return (x,seq_len),y

    def __next__(self):
        if self.residue and self.index == self.n_batches: # 针对剩余的不足以构成一个batch的数据
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration  # 代表停止迭代
        else:
            batches = self.batches[self.index * self.batch_size : (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter

def get_time_dif(start_time):
    t = time.time()
    running_time = t - start_time
    return round(running_time)

if __name__ == '__main__':
    dataset = 'THUCNews'
    train_path = os.path.join(dataset, 'NEWS/train.txt')
    tokens,labels = load_file(train_path,False)
    print(tokens[:10])
