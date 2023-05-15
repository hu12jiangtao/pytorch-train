import numpy as np
from torch import nn
import torch
from einops import repeat, rearrange
from Transformer import TransformerBlock
from timm.models.layers import trunc_normal_
from einops.layers.torch import Rearrange
from timm.models.vision_transformer import Block

# 在原始代码中的transformer的模块利用的是Block(emb_dim, num_head),除了前两个参数其余都是默认参数
# 在原始代码中的transformer的dropout都是0，而自己写的则是dropout=0.1(此时导致验证集的准确率出现问题？)
# 但是在原始的代码中的优化器采用的是带有正则化的Adam优化器，因此在transformer的block中不在设置正则化？

def random_indexes(T):
    forward_indexes = list(np.arange(T))
    np.random.shuffle(forward_indexes) # forward_index[i]为代表随机打乱后第i个patch是实际的patch的索引
    backward_indexes = np.argsort(forward_indexes) # backward_indexes[i]代表原先patch的索引为i的patch在随机打乱后得到的索引
    return forward_indexes,backward_indexes

def take_indexes(patches,indexes): # patches=[patches_num,batch,embedding_size]
    return torch.gather(patches,dim=0,index=repeat(indexes,'seq batch -> seq batch embedding',embedding=patches.shape[-1]))

class PatchShuffle(nn.Module):
    def __init__(self,ratio):
        super(PatchShuffle, self).__init__()
        self.ratio = ratio
    def forward(self,patches): # [patches_num, batch,embedding_size]
        patches_num, batch, embedding_size = patches.shape
        remain_T = int((1 - self.ratio) * patches_num) # 每张图片没被遮掩的图片的数量
        indexes = [random_indexes(patches_num) for _ in range(batch)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes],axis=-1),dtype=torch.long,
                                          device=patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes],axis=-1),dtype=torch.long,
                                           device=patches.device) # [patch_size,batch]
        # 此时将patch放在最前面主要是因为gather需要随机抽取每个批量的每个随机打乱的patch特征向量
        # 此时需要将原先的patch的顺序进行打乱
        # forward_indexes中的值全部都是小于patches_num的
        # 此时输出out[0][1][0] = patches[indexes[0][1][0]][1][0]=patches[k][1][0]
        # 代表的是输出的第1个batch的第0个patch来自于输入的第1个batch的第k个patch
        # 此时等式左边输出的patches为原先随机打乱的patches索引
        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]
        return patches,forward_indexes,backward_indexes

class MAE_Encoder(torch.nn.Module): # 将随机抽取的patches输入到ViT中（此时批量放在第一维上）
    def __init__(self,image_size=32,patch_size=2,emb_dim=192,num_layer=12,num_head=3,mask_ratio=0.75):
        super(MAE_Encoder, self).__init__()
        self.patchify = nn.Conv2d(3,emb_dim,kernel_size=patch_size,stride=patch_size) # 利用与将二维的图片转换为一维的序列
        self.shuffle = PatchShuffle(mask_ratio)
        self.cls_token = nn.Parameter(torch.zeros(size=(1,1,emb_dim))) # 此时其中cls_token中已经包含了位置信息
        self.position_embedding = nn.Parameter(torch.zeros(size=((image_size // patch_size)**2,1,emb_dim))) # 输入patches的位置信息
        # self.transformer = nn.Sequential(*[TransformerBlock(emb_dim, num_head,
        #                                  dim_head=64, mlp_dim= 4 * emb_dim, dropout=0.1) for _ in range(num_layer)])
        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token,std=0.02)
        trunc_normal_(self.position_embedding,std=0.02)

    def forward(self,images): # image.shape=[batch,3,height,weight]
        patches = self.patchify(images) # [batch,emb_size,h,w]
        patches = rearrange(patches,'b e h w -> (h w) b e ') # [patch_num, batch, embedding_size]
        patches = patches + self.position_embedding
        # 此时的patches为未被遮掩的每个batch的patches，patches.shape=[remain_T,batch,emb_size]
        # 此时forward_indexes=[patches_num,batch],backward_indexes=[patches_num,batch]
        # 其中forward_indexes[0][1]中存储着 打乱后的第1个batch中的第0个patch实际是第1个batch中的第forward_indexes[0][1]个patch
        # backward_indexes[:][1]中存储着在第1个batch中实际的patches顺序在forward_indexes中的索引位置
        patches,forward_indexes,backward_indexes = self.shuffle(patches)
        # 构建ViT的输入patches.shape=[patches_num + 1, batch, embedding_size]
        patches = torch.cat([self.cls_token.expand(self.cls_token.shape[0],
                                                   patches.shape[1],self.cls_token.shape[-1]),patches],dim=0)
        patches = rearrange(patches,'t b e -> b t e') # [batch, patches_num + 1, embedding_size]
        features = self.layer_norm(self.transformer(patches)) # 得到未被遮掩的patches的特征向量
        features = rearrange(features,'b t e -> t b e') # [patches_num + 1, batch, embedding_size]
        return features,backward_indexes # 此时出现错误，应当时features[1:],原因是在decode中不需要利用到语意特征向量
        # 这里出现了问题但是不高改，改了话MAE的预训练模型的参数都发生了改变(在之后的练习项目中进行了修正)

class MAE_Decoder(torch.nn.Module):  # 构建一个轻量级的解码器
    def __init__(self,image_size=32,patch_size=2,emb_dim=192,num_layer=4,num_head=3):
        super(MAE_Decoder, self).__init__()
        # 此时的输入应当是所有的patches(此时的被遮掩的patches共用一个特征向量，同时输入还有一个位置矩阵)
        self.mask_token = nn.Parameter(torch.zeros(1,1,emb_dim))
        # decode输入的位置矩阵
        self.position_embedding = nn.Parameter(torch.zeros(size=((image_size // patch_size)**2, 1, emb_dim)))
        # decode的模型
        # self.transformer = nn.Sequential(*[TransformerBlock(emb_dim, num_head,
        #                                  dim_head=64, mlp_dim= 4 * emb_dim, dropout=0.1)for _ in range(num_layer)])
        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])
        # 将所有的patches在decode输出的向量转换为所有的像素点
        self.heads = nn.Linear(emb_dim, 3 * patch_size ** 2)
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)',
                                   h=image_size//patch_size,p1=patch_size,p2=patch_size)

    def forward(self, features, backward_indexes):
        remain_T = features.shape[0]
        # 输入的feature为没有被mask的patch的输出维度,[remain_T,batch,embedding_size]
        # 被mask的patches共用一个向量，此时的形状为[patch_num - remain_T, batch, embedding_size]
        mask_feature = self.mask_token.expand(backward_indexes.shape[0] - features.shape[0],features.shape[1],-1)
        # 合成整个和forward_index所对应的特征矩阵,此时forward_indexes[i][j]中存储的值为随机打乱顺序后第j个batch的第i个patch
        # 此时的feature[i][j]存储这对应着forward_indexes[i][j]的特征向量
        features = torch.cat([features,mask_feature],dim=0)
        # backward_indexes[0][i]中存储着第i个batch(第i张图中)，在没有被打乱的情况下第i张图片的第0个patch为打乱顺序后forward_index[:][i]上的索引
        # 若此时的索引为j，对于实际的第i张图片的第0个patch的向量应当为forward_index[backward_indexes[j][i]][i]对应的特征向量
        # 因此此时为backward_indexes[0][i]对应的应当是features[[backward_indexes[j][i]][i][:]]
        # 此时out[j][i][k] = features[[backward_indexes[j][i]][i][k]]
        # 与gather函数out[j][i][k] = features[[backward_indexes[j][i][k]][i][k]]进行比较
        # 此时对于任意的k来说gather中的backward_indexes[j][i][k]的值 = 原先backward_indexes[j][i]
        # 因此gather中的backward_indexes = repeat(backward_indexes,'t b' -> 't b e',e=embedding_size)
        features = take_indexes(features, backward_indexes)
        # 加入位置矩阵
        features = features + self.position_embedding # [patch_num,batch,embedding_size]
        features = rearrange(features,'t b e -> b t e')
        # 输入解码器(此时得到的是按照每张图片的patches顺序排列的特征向量)
        features = self.transformer(features) # [batch,patch_num,embedding_size]
        features = rearrange(features, 'b t e -> t b e') # [patch_num,batch,embedding_size]
        # 转换为每个patch的像素
        patches = self.heads(features) # [patch_num,batch,patch_size * patch_size * 3]
        # 计算损失的时候只考虑那些被遮掩的patches生成的像素
        mask = torch.zeros_like(patches)
        mask[remain_T:] = 1
        # 此时的mask应当和forward_index相对应
        # 即forward_index[:][i]的前remain_T个序列的patches进入encode，因此mask[:][i]的前remain_T个序列的patches的mask的系数为0
        # 由于对于第i张图片来说backward_indexes[0][i]代表第i张图片的原先的第0个patch在forward_index上的索引
        # 若此时backward_indexes[0][i]<remain_T的时候mask[backward_indexes[0][i]][i]的值应当为0
        # 因此此时mask[j][i] = mask[backward_indexes[j][i]][i]
        # 此时的mask对于每张图片来说对于经过encode的patches的向量的值都是0
        mask = take_indexes(mask,backward_indexes)
        mask = self.patch2img(mask) # 其中元素为0代表需要遮掩的部分(输入encode的patches)
        # 转换为一张图片
        img = self.patch2img(patches) # img.shape=[batch,3,height,weight]
        return img,mask

class MAE_ViT(torch.nn.Module): # 整个完整的MAE模型(非对称的encode-decode)
    def __init__(self,image_size,patch_size,emb_dim,encode_num_layer,
                 encode_num_head,decode_num_layer,decode_num_head,mask_ratio):
        super(MAE_ViT, self).__init__()
        self.encode = MAE_Encoder(image_size,patch_size,emb_dim,encode_num_layer,encode_num_head,mask_ratio)
        self.decode = MAE_Decoder(image_size,patch_size,emb_dim,decode_num_layer,decode_num_head)

    def forward(self,images):
        features,backward_indexes = self.encode(images)
        predict_images,mask = self.decode(features,backward_indexes)
        return predict_images,mask

class ViT_Classifier(torch.nn.Module): # 用于之后的迁移
    def __init__(self,encode,num_classes):
        super(ViT_Classifier, self).__init__()
        self.position_embedding = encode.position_embedding
        self.cls_token = encode.cls_token # [1,1,embedding_size]
        self.patchify = encode.patchify
        self.transformer = encode.transformer
        self.layer_norm = encode.layer_norm
        self.head = nn.Linear(self.cls_token.shape[-1], num_classes)

    def forward(self,images):
        patches = self.patchify(images) # [b,c,h,w]
        patches = rearrange(patches,'b c h w -> (h w) b c ') # [patch_num,batch,embedding_size]
        patches = patches + self.position_embedding # 加入了位置信息
        patches = torch.cat([self.cls_token.expand(-1,patches.shape[1],-1),patches],dim=0) # [patch_num + 1,batch,embedding_size]
        patches = rearrange(patches,'e b c -> b e c')
        features = self.layer_norm(self.transformer(patches)) # features = [batch, patch_num + 1,embedding_size]
        features = features[:,0] # [batch,embedding_size]
        logits = self.head(features) # [batch, 10]
        return logits














