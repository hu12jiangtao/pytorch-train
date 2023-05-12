# word2vec 的功能 和 torch中embedding相类似，都是将一个单词或者字符转换为低维的向量进行表示
# 两者的区别是word2vec是一个无监督的学习过程(在训练过程中没有用到训练集的标签部分，仅看输入文本)，
# 而embedding是一个有监督的学习(利用到训练集中的输出文本，会根据target进行学习和调整)
# 因此Word2vec一般单独提前训练好，而Embedding一般作为模型中的层随着模型一同训练

# 对于Word2vec的训练一般存在两种方式:
# 方式1:CBOW(通过前后几个单词来预测当前出现的值)，此时的训练方法和SKIP_GROW的训练方法相同，只是测试时不同？

# 方式2:SKIP_GROW模型(通过当前的单词来预测前后可能出现的单词)，从而获取每个单词的映射向量


# word2vec的skip_grow案例的训练样本: 今天 天气 真好 ，其中embed_size = 50, vocab_size = 5000(之后代码所实现的)
# word2vec的skip_grow的目标为:通过当前的给定的词来预测出现的下一个词(给定天气，预测真好,n_gram规定了可能出现的词是周围的哪几个词)，
# 第一步:
# 首先将给定token_to_idx将此进行序列化后再将其进行独热化得到一个1*5000的向量，经过一个全连接变换为了[1,50]的向量（就是nn.Embedding的工作）
# 此时的训练参数为shape=[5000,50]矩阵,由于1*5000的向量是一个独热向量，因此相当于取出训练参数的i行(输入词语的序列为i)
# 第二步:
# 将映射后的[1,50]的向量输入一个训练参数为[50,5000]为训练参数的全连接层并且经过softmax得到一个[1,5000]的向量代表每一个词出现的概率
# 之后利用交叉熵函数求解损失并且进行训练
from torch import nn
import pickle
import pandas as pd
import jieba
import torch
from tqdm import tqdm

def load_stop_words(file = "stopwords.txt"):
    with open(file,'r',encoding='utf-8') as f:
        return f.read().split('\n')

def cut_words(file="数学原始数据.csv"):
    stop_words = load_stop_words()
    result = []
    all_data = pd.read_csv(file,'gbk',names=['NEWS'])['NEWS']
    for words in all_data:
        c_words = jieba.lcut(words) # 让他从左边自动进行分词，返回一个列表
        result.append([word for word in c_words if word not in stop_words])
    return result

def get_dict(data):
    idx_to_token = []
    for sentence in data:
        for word in sentence:
            if word not in idx_to_token:
                idx_to_token.append(word)
    token_to_idx = {word:index for index,word in enumerate(idx_to_token)}
    vocab_size = len(idx_to_token)
    word_to_onehot = {}
    for index,word in enumerate(idx_to_token):
        one_hot = torch.zeros(size=(1,vocab_size),device=device)
        one_hot[0,index] = 1
        word_to_onehot[word] = one_hot
    return token_to_idx,idx_to_token,word_to_onehot

class My_net(nn.Module):
    def __init__(self,vocab_size,embedding_size):
        super(My_net, self).__init__()
        self.linear1 = nn.Linear(vocab_size,embedding_size,bias=False)
        self.linear2 = nn.Linear(embedding_size,vocab_size,bias=False)
    def forward(self,x):
        return self.linear2(self.linear1(x))

class add_machine():
    def __init__(self,n):
        self.data = [0.] * n
    def add(self,*args):
        self.data = [i + float(j) for i,j in zip(self.data,args)]
    def __getitem__(self, item):
        return self.data[item]


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = cut_words()
    token_to_idx,idx_to_token,word_to_onehot = get_dict(data)
    vocab_size = len(idx_to_token)
    embedding_size = 107
    sum_epoch = 20
    n_gram = 3
    net = My_net(vocab_size,embedding_size)
    net.to(device)
    loss = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters())
    for epoch in range(sum_epoch):
        for sentence in tqdm(data):
            for index,word in enumerate(sentence):
                now_word_onehot = word_to_onehot[word]
                other_words = sentence[max(0,index - n_gram):index] + \
                              sentence[index + 1: index+1+n_gram] # 其为当前单词的前后三个单词
                for other_word in other_words:
                    y_hat = net(now_word_onehot)
                    label = torch.tensor([token_to_idx[other_word],],dtype=torch.long,device=device)
                    l = loss(y_hat,label)
                    opt.zero_grad()
                    l.backward()
                    opt.step()
        print(f'loss:{l:1.4f}')
        print('*'*50)
    w1 = net.get_parameter('linear1.weight')
    with open("word2vec.pkl","wb") as f:
        pickle.dump([w1,token_to_idx,idx_to_token],f)






