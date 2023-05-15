# 此处就是写出一个transform的block块(按照之前的ViT来)
from torch import nn
import torch
import math
from torch.nn import functional as F

class DotAttention(nn.Module):
    def __init__(self,dropout):
        super(DotAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
    def forward(self,query,key,value):
        query_size = query.shape[-1]
        weight = torch.bmm(query,key.permute(0,2,1)) / math.sqrt(query_size) # weight=[b,num_q,num_k]
        return torch.bmm(F.softmax(weight,dim=-1), value)

class LSA(nn.Module):  # 自注意力机制
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super(LSA, self).__init__()
        self.heads = heads
        inner_dim = heads * dim_head
        self.w_q = nn.Linear(dim, inner_dim)
        self.w_k = nn.Linear(dim, inner_dim)
        self.w_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim),nn.Dropout(dropout))
        self.attention = DotAttention(dropout)
    def transpose_qkv(self,x): # x=[batch, seq, heads * dim_head]
        x = x.reshape(x.shape[0],x.shape[1],self.heads,-1)
        x = x.permute(0,2,1,3)
        x = x.reshape(-1,x.shape[2],x.shape[3])
        return x
    def transpose_out(self,x): # x=[batch * head,seq,dim_head]
        x = x.reshape(-1,self.heads,x.shape[1],x.shape[2])
        x = x.permute(0,2,1,3)
        x = x.reshape(x.shape[0],x.shape[1],-1)
        return x
    def forward(self,x):
        query = self.transpose_qkv(self.w_q(x))
        key = self.transpose_qkv(self.w_k(x))
        value = self.transpose_qkv(self.w_v(x))
        out = self.attention(query,key,value) # [batch * head, seq, dim_head]
        out = self.transpose_out(out) # [batch, seq, dim_head * head]
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(nn.Linear(dim,hidden_dim),nn.GELU(),nn.Dropout(dropout),
                                 nn.Linear(hidden_dim,dim),nn.Dropout(dropout))
    def forward(self,x):
        return self.net(x)

class AddNorm(nn.Module):
    def __init__(self,dim):
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
    def forward(self,x,y):
        return x + self.norm(y)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super(TransformerBlock, self).__init__()
        self.norm1 = AddNorm(dim)
        self.norm2 = AddNorm(dim)
        self.FFN = FeedForward(dim, mlp_dim, dropout)
        self.attention = LSA(dim, heads, dim_head, dropout)
    def forward(self,x):
        out = self.norm1(x,self.attention(x))
        out = self.norm2(out,self.FFN(out))
        return out


