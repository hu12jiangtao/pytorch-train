from torch import nn
import torch
from einops.layers.torch import Rearrange
from torch.nn import functional as F
import math

class SPT(nn.Module): # 获得每个patch进行embedding后的结果
    def __init__(self,dim,patch_size,channel=3):
        patch_dim = patch_size * patch_size * 5 * channel
        super(SPT, self).__init__()
        self.to_patch_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',p1=patch_size,p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim)
        )

    def forward(self,x): # x.shape=[b,c,h,w]
        offsets = ((1,-1,0,0),(-1,1,0,0),(0,0,1,-1),(0,0,-1,1))
        offsets_x = list(map(lambda offset: F.pad(x,offset),offsets))

        x_with_offset = torch.cat([x, *offsets_x],dim=1) # x_with_offset.shape=[b,5*c,h,w]
        return self.to_patch_tokens(x_with_offset) # [b, h_num * w_hum, patch_size * patch_size * c * 5] -> [b, h_num * w_hum, dim]

def pair(x):
    return x if isinstance(x,tuple) else (x,x)

class DotAttention(nn.Module): # 此时query_size=key_size
    def __init__(self,dropout):
        super(DotAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self,query,key,value):
        query_size = query.shape[-1]
        weight = torch.bmm(query,key.permute(0,2,1)) / math.sqrt(query_size) # [b,num_q,num_k]
        return torch.bmm(F.softmax(weight,dim=-1),value)

class LSA(nn.Module): # 多头注意力机制层(此时无掩码操作),且其为自注意力机制(输入key=query=value)
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super(LSA, self).__init__()
        # dim为多头注意力机制输入、输出的维度，此时为256
        self.attention = DotAttention(dropout)
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        self.to_query = nn.Linear(dim, inner_dim)
        self.to_key = nn.Linear(dim, inner_dim)
        self.to_value = nn.Linear(dim, inner_dim)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def transpose_qkv(self,x): # [batch,seq,hidden_num]
        x = x.reshape(x.shape[0],x.shape[1],self.heads,self.dim_head)
        x = x.permute(0,2,1,3)
        x = x.reshape(-1,x.shape[-2],x.shape[-1])
        return x # [batch * heads, seq, head_dim]

    def transpose_out(self,x):
        x = x.reshape(-1,self.heads,x.shape[1],x.shape[2])
        x = x.permute(0,2,1,3)
        x = x.reshape(x.shape[0],x.shape[1],-1)
        return x # [batch, seq, head_dim * heads]

    def forward(self,x):
        query = self.transpose_qkv(self.to_query(x))
        key = self.transpose_qkv(self.to_key(x))
        value = self.transpose_qkv(self.to_value(x))
        out = self.attention(query,key,value) # out = [batch * heads, seq, head_dim]
        out = self.transpose_out(out) # [batch, seq, head_dim * heads]
        return self.to_out(out)

class AddNorm(nn.Module): # AddNorm中的线性归一化层
    def __init__(self,dim):
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self,x, y):
        return x + self.norm(y)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(nn.Linear(dim,hidden_dim),
                                 nn.GELU(),nn.Dropout(dropout),
                                 nn.Linear(hidden_dim,dim),
                                 nn.Dropout(dropout))
    def forward(self,x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super(TransformerBlock, self).__init__()
        self.norm1 = AddNorm(dim)
        self.norm2 = AddNorm(dim)
        self.attention = LSA(dim, heads, dim_head, dropout)
        self.FFN = FeedForward(dim, mlp_dim, dropout)

    def forward(self,x):
        out = self.norm1(x, self.attention(x))
        out = self.norm2(out, self.FFN(out))
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super(Transformer, self).__init__()
        self.net = nn.ModuleList()
        for i in range(depth):
            self.net.append(TransformerBlock(dim, heads, dim_head, mlp_dim, dropout))

    def forward(self,x):
        for blk in self.net:
            x = blk(x)
        return x

class ViT_small(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,
                 pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super(ViT, self).__init__()
        # 此时的dim就是embedding后的维度(encode的输入维度)
        image_height, image_weight = pair(image_size)
        patch_height, patch_weight = pair(patch_size)
        assert image_height % patch_height == 0 and image_weight % image_weight == 0
        num_patches = (image_height // patch_height) * (image_weight // patch_weight)
        self.cls_token = nn.Parameter(torch.randn(size=(1,1,dim)))
        self.to_patch_embedding = SPT(dim, patch_size, channels)
        self.pos_embedding = nn.Parameter(torch.randn(size=(1, 1+num_patches, dim))) # [batch,1+num_patches,dim]
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,num_classes))

    def forward(self,images): # [batch,c,H,W]
        x = self.to_patch_embedding(images) # [batch,num_patches,dim]
        batch, seq = x.shape[0], x.shape[1]
        cls_token = self.cls_token.repeat((batch, 1, 1)) # [batch,1,dim]
        x = torch.cat([cls_token,x],dim=1)
        x += self.pos_embedding[:,: seq + 1]
        x = self.dropout(x) #
        x = self.transformer(x) # [batch,1 + seq,dim]
        x = x.mean(dim=1) if self.pool == 'mean' else x[:,0] # [batch,dim]
        return self.mlp_head(x)


class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,
                 pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super(ViT, self).__init__()
        # 此时的dim就是embedding后的维度(encode的输入维度)
        image_height, image_weight = pair(image_size)
        patch_height, patch_weight = pair(patch_size)
        assert image_height % patch_height == 0 and image_weight % image_weight == 0
        num_patches = (image_height // patch_height) * (image_weight // patch_weight)
        patch_dim = channels * patch_height * patch_weight
        self.cls_token = nn.Parameter(torch.randn(size=(1,1,dim)))
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_weight),
            nn.Linear(patch_dim, dim),
        ) # 此处和VIT_small不相同
        self.pos_embedding = nn.Parameter(torch.randn(size=(1, 1+num_patches, dim))) # [batch,1+num_patches,dim]
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,num_classes))

    def forward(self,images): # [batch,c,H,W]
        x = self.to_patch_embedding(images) # [batch,num_patches,dim]
        batch, seq = x.shape[0], x.shape[1]
        cls_token = self.cls_token.repeat((batch, 1, 1)) # [batch,1,dim]
        x = torch.cat([cls_token,x],dim=1)
        x += self.pos_embedding[:,: seq + 1]
        x = self.dropout(x) #
        x = self.transformer(x) # [batch,1 + seq,dim]
        x = x.mean(dim=1) if self.pool == 'mean' else x[:,0] # [batch,dim]
        return self.mlp_head(x)








