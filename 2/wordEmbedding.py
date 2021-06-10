#################################
#调用PyTorch包
#################################
#基本上所有torch脚本都需要用到
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud #Pytorch读取训练集需要用到torch.utils.data类


#################################
#调用其他包
#################################
from collections import Counter
import numpy as np
import random
import math

import pandas as pd
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity


#################################
#初始设置
#################################
#调用gpu
USE_CUDA=torch.cuda.is_available()

#为保证实验结果可以浮现，将各种random seed固定到一个特定的值
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
if USE_CUDA:
    torch.cuda.manual_seed(1)
    
#设定一些hyper parameters
C=3 #nearby words threshold 指定前后3个单词进行预测
K=100 #number of negative samples 负样本随机采样数量；每一个正样本对应K个负样本
NUM_EPOCHS=2 #The num of epochs of training 迭代轮数
MAX_VOCAB_SIZE=30000 #the vocabulary size 词汇表大小
BATCH_SIZE=128
LEARNING_RATE=0.2 #the initial learning rate
EMBEDDING_SIZE=100 #词向量维度

#tokenize函数 将文本转化为一个个单词
def word_tokenize(text):
    return text.split()

#################################
#数据预处理及相关操作
#################################
#读取文件
with open('./text8/text8.train.txt','r') as fi:
    text=fi.read()
    
# len(text)

#分词
#str.lower()将str中大写转化为小写
text=[w for w in word_tokenize(text.lower())]

#将出现频率最高的MAX_VOCAB_SIZE-1个单词取出来，以字典的形式存储(包含每个单词出现次数)
#-1留给UNK单词
#collection.Counter(text): 计算每个元素出现个数 返回counter对象
#Counter(text).most_common(N): 找到text中出现最多的前N个元素
#https://zhuanlan.zhihu.com/p/350899229
vocab=dict(Counter(text).most_common(MAX_VOCAB_SIZE-1))
#将UNK单词添加进vocab
#UNK出现次数=总单词出现次数-常见单词出现次数
#dic.values() 返回字典中所有值所构成的对象
vocab['<unk>']=len(text)-np.sum(list(vocab.values()))

#从vocab中取出所有单词
idx_to_word=[word for word in vocab.keys()]

#以字典的形式取得单词及其对应的索引
#enumerate: 接收一个可遍历的数据对象['a','b','c'] 返回索引与对象的组合[(0,'a'),(1,'b'),(2,'c')]
#索引值与单词出现次数相反，最常见单词索引为0。
word_to_idx={word:i for i,word in enumerate(idx_to_word)}

# list(word_to_idx.items())[:100]

#计算每个单词频率 负采样时需要使用
#获得所有单词出现的次数
word_counts=np.array([count for count in vocab.values()], dtype=np.float32)
#计算所有单词的频率
word_freqs=word_counts/np.sum(word_counts)
#论文Distributed Representations of Word...中频率取了3/4次方
word_freqs=word_freqs**(3./4.)
#重新normalize 重新计算所有单词频率 类似softmax
word_freqs=word_freqs/np.sum(word_freqs)

#检查单词数为MAX_VOCAB_SIZE
VOCAB_SIZE=len(idx_to_word)


#################################
# 实现Dataloader
#################################
class WordEmbeddingDataset(tud.Dataset):
    def __init__(self, text, word_to_idx, idx_to_word, word_freqs, word_counts):
        #初始化模型
        #super(WordEmbeddingDataset, self).__init__()
        super().__init__()
        
        #顺序存储每个在text中word的word_to_idx中的索引(序号)，
        #如果word不在word_to_idx中（属于unk）则存储unk在word_to_idx中对应的序号
        self.text_encoded=[word_to_idx.get(word,word_to_idx['<unk>']) for word in text]
        #转化为int型LongTensor
        self.text_encoded=torch.LongTensor(self.text_encoded)
        
        #将输入的参数初始化为torch tensor
        self.word_to_idx=word_to_idx
        self.idx_to_word=idx_to_word #类中没有使用
        self.word_freqs=torch.Tensor(word_freqs)
        self.word_counts=torch.Tensor(word_counts) #类中没有时使用
        
    #数据集一共有多少个item
    def __len__(self):
        return len(self.text_encoded)
    
    #提供一个index 返回一串训练数据
    #index为训练数据集中每个单词对应的序号,即text_encoded中每个元素下标
    def __getitem__(self, index):
        #中心词 根据index可获得text中index位置的词(以数字表示)
        center_word=self.text_encoded[index]
        
        #周围词 为中心词前C个词与后C个词
        #pos_indices_list存储了中心词的周围词对应的序号
        #注意当index=0,1,2, len(self.text_encoded)-3,len(self.text_encoded)-2,len(self.text_encoded)-1时,
        #pos_indices_serialNumber的范围会超出text_encoded的范围
        pos_indices_serialNumber=list(range(index-C,index))+list(range(index+1,index+1+C))
        #print(pos_indices_serialNumber)
        
        #所以需要对pos_indices_serialNumber中的元素逐个同text_encoded的长度取余,
        #个人认为这一步的合理性存在疑问
        #将训练集最后的几个词作为最开始几个中心词的周围词/将训练集最初的几个词作为最后几个中心词的周围词
        #都没有合理性
        pos_indices_new_serialNumber=[i % len(self.text_encoded) for i in pos_indices_serialNumber]
        #print(pos_indices_new_serialNumber)
        #print(type(pos_indices_new_serialNumber))
        
        #由pos_indices_new_serialNumber获得text中对应位置的词(以数字表示)
        #text_encoded为Tensor,可以接收一组数组作为序号,返回序号对应的元素
        pos_words=self.text_encoded[pos_indices_new_serialNumber]
        #print(type(self.text_encoded))
        #print(pos_words)
        
        #用于negative sampling
        #参考https://towardsdatascience.com/nlp-101-negative-sampling-and-glove-936c88f3bc68
        
        #torch.multinomial
        #https://pytorch.org/docs/stable/generated/torch.multinomial.html
        #作用是对self.word_freqs做K * pos_words.shape[0]次取值，输出的是self.word_freqs对应的下标。
        #取样方式采用有放回的采样，并且self.word_freqs数值越大，取样概率越大。
        #每个正确的单词采样K个，pos_words.shape[0]是正确单词数量,pos_words.shape[0]的值为6
        neg_words=torch.multinomial(self.word_freqs, K*pos_words.shape[0], True)
        #print(neg_words)
        
        return center_word, pos_words, neg_words


#################################
#定义Dataset, dataloader
#################################
dataset=WordEmbeddingDataset(text, word_to_idx, idx_to_word, word_freqs, word_counts)

#num_workers: 线程数量
dataloader=tud.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

#测试
# next(iter(dataloader))


#################################
#定义PyTorch模型
#################################

class EmbeddingModel(nn.Module):
    #定义网络架构所需参数
    #初始化输入和输出embedding
    def __init__(self, vocab_size, embed_size):
        super().__init__()

        self.vocab_size=vocab_size #30000
        self.embed_size=embed_size #100
        
        #定义in和out两个embedding层 in_embed和out_embed相当于参数w
        #https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        #nn.Embedding通常用于word embedding(https://www.zhihu.com/question/32275069/answer/80188672)
        #此处输出(30000,100)的embedding
        self.in_embed=nn.Embedding(self.vocab_size, self.embed_size)
        self.out_embed=nn.Embedding(self.vocab_size, self.embed_size)
    
    #定义forward函数(定义网络架构)
    #*********************************此处需要进一步分析理解
    # input_labels: 中心词  (batch_size)个单词
    # pos_labels: 正确的周围词  (batch_size,(window_size*2))个单词
    # neg_labels: 负采样中选取的错误的周围词  (batch_size, (window_size*2*K))个单词
    def forward(self, input_labels, pos_labels, neg_labels):

        #这里进行了运算：（batch_size,vocab_size）*（vocab_size,embed_size）= 128(B) * 100 (embed_size)
        input_embedding=self.in_embed(input_labels) #(batch_size, embed_size)
        pos_embedding=self.out_embed(pos_labels) #(batch_size,(window_size*2), embed_size)
        neg_embedding=self.out_embed(neg_labels) #(batch_size, (window_size*2*K), embed_size)

        #a.unsqueeze(n) 在a的第n维增加一个
        input_embedding=input_embedding.unsqueeze(2) #(batch_size, embed_size, 1)
        
        #计算中心词embedding与周围词embedding的乘积
        #bmm: If input1 is a (b×n×m) tensor, input2 is a (b×m×p) tensor, out will be a (b×n×p) tensor.
        #https://pytorch.org/docs/stable/generated/torch.bmm.html
        #squeeze() 删除维度为1的维度
        pos_dot=torch.bmm(pos_embedding, input_embedding).squeeze() #(batch_size, (window_size*2))
        #计算中心词embedding与错误周围词embedding的乘积
        neg_dot=torch.bmm(neg_embedding, input_embedding).squeeze() #(batch_size, (window_size*2*K))
        
        #论文'Distributed Representations of Words and Phrases and their Compositionality'中第3页末尾公式
        #计算加号前'中心词embedding与周围词embedding的乘积'的logsigmoid
        log_pos=F.logsigmoid(pos_dot)
        #计算加号后'中心词embedding与错误周围词embedding的乘积'的logsigmoid
        log_neg=F.logsigmoid(-neg_dot)
        #忽略Ewi∼Pn(w)的部分(此操作已在定义neg_words时完成操作)
        #对加号后面部分log_neg求和
        log_neg=log_neg.sum(1)
        #这一步没有出现在公式中...
        log_pos=log_pos.sum(1)

        loss=log_pos+log_neg

        return -loss

        #取出input_embeddings
        def input_embeddings(self):
            return self.in_embed.weight.data.cpu.numpy()


#实例化模型
model=EmbeddingModel(VOCAB_SIZE,EMBEDDING_SIZE)

#使用cuda
if USE_CUDA:
    model=model.cuda()

