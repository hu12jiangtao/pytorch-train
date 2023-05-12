import torch
import pickle

w1,token_to_idx,idx_to_token = pickle.load(open('word2vec20.pkl','rb'))

def word_voc(word,matrix):  # 用于得到一个单词对应的特征向量
    return matrix[:,token_to_idx[word]]

def voc_sim(word, top_n):
    v_w1 = word_voc(word,w1)
    word_sim = {}
    for i in range(len(token_to_idx)):
        v_w2 = word_voc(idx_to_token[i],w1)
        theta_sum = torch.sum(v_w2 * v_w1)
        theta_den = torch.norm(v_w2) * torch.norm(v_w1)
        theta = theta_sum / theta_den
        word = idx_to_token[i]
        word_sim[word] = theta.item()
    word_sim = sorted(word_sim.items(),key=lambda x:x[1],reverse=True)
    for word,sim in word_sim[:top_n]:
        print(word,sim)



voc_sim(word = '分子', top_n = 20)
