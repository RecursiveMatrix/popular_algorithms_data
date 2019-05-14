# 词袋模型：把每个单词用热编码形式表示，长度为所有词组成词典的长度。
# 这样对于某一句话，可以统计出每个词的个数，然后用一个近似统计的方式展示出来
# 例如 单词'watch'= 0，0，1 某一句话中出现来3次'watch' 可表示为0，2，3
# 这样对于问题是：维度灾难，没有考虑语序，无法表现同义词


# 语句模型：某一个句子W,有T个词，分别w1，w2。。wT, w[i,j]表示句子中从第i个词到第j个词的部分
# 所有这个句子按照这个方式排列的概率为：
# P(W) = P(W[1,T]) = P(w1) * P(w2|w1) * P(w3|w[1,2]) *...* P(wT|w[1,T])
# 所以问题转化为求 P(w1) 和 P(wk|w[1,k-1])，后者可根据 n-gram模型和神经网络模型两种方式计算


# n-gram模型：
# 根据贝叶斯公式，P(wi|w[1,i-1]) = P（w[1,i]) / P(w[1,i-1])，即某个词在当前位置出现的概率 = 这句话组成的概率/这句话除了当前位置词之前句子组成的概率
# 通过大数定律，用频数代替频率。P（w[1,i]) 大致为 N（w[1,i])，即 w[1,i]在语言库中出现的次数
# 由于当i很大时，计算 N（w[1,i])很费时。 n-gram假设一个词的出现之和它前面固定数目的词（即n个词）有关系，做了一个n-1阶的Markov的假设。
# 假设： P(wi|w[1,i-1]) = P(wi|w[i-1,i-n+1]) = P（w[i,i-n+1]) / P(w[i-1,i-n+])


# 神经语言网络模型：
# 神经网络分为三层：
# 第一层输入为词序列，（每个词有一个index，用来表示顺序）非线性变化到第二层，tanh激活后进入第三层，非线性后 softmax 输出结果 P(wi|w[1,i-1])
# 设输入为x，x=（C(wt-n+1),...C(wt-2),C(wt-1)） y = softmax(b+Wx+Utanh(d+Hx))
# 上式中 ： 将第t个词的index通过矩阵 C（不同词之间参数共享）来转化成向量，作为输入x
# x一方面通过神经网络直接连入最后一层，为Wx
# 另一方面同时连入隐藏层，为Hx，再经过激活后连入最后一层，为tanh（d+Hx）
# 以上两部分经过softmax后输出
# 最终希望的部分是存储在 C 中的词向量




# intro: word2vec 是一个自然语言处理的模型，就是把一个word变成一个vector，对于中文word可以是一个词或者一个单词
# 将词变成向量的好处：
    # 降维： 否则通过热编码，维度会非常大；相反，如果把每个词都表示为长度为n的向量，就可以避免这个麻烦
    # 可以挖掘词之间的联系： 如果样本足够大，有些词会有更近的联系，如猫和喵的联系比猫和狗更紧密，向量'queen'-向量'female'=向量'king'-向量'male'

# word2vec模型是一个只有一个隐藏层的神经网络，每个词是一条数据，目的是预测这个词的后一个或者后k个词
# 输入数据是热编码的向量，输出的结果是预测这个词后面出现的词
# 该模型中的向量，就是隐藏层的数值

# CBOW：与神经网络不同，去掉了隐藏层
# 取用指定的window值，也就是目标词前后各n个词，例如window=2，就是前后各两个词
# 所以此时预测的是（wt|wt-k，wt-（k-1）。。。wt+k）
# 输入为前后k个一共2k个词的one-hot表示，维度都为v*1，v为所有词的个数
# 记输入层到隐藏层的权重为，维度W*d，d为给定词向量的维度，即一句话多少词
# 隐藏层向量为h，维度d*1
# h的计算为权重矩阵W，分别与x向量相乘，求和，再取平均
# 隐藏到输出的权重U， 维度是 d*V，输出y，维度 V *1
# 虽然输出向量y和x维度相同，但是y不是one-hot code，向量y中输出的是每个词的可能性
# 所以cost function 是需要让目标词的可能性最大
# 但是softmax需要遍历所有词表，所以后续用hierarchical softmax

# skip-gram
# 预测到是在当前词是wt的情况下，之前之后k个词的概率

# hierarchical softmax 是一种huffman树
# 树的叶子节点是训练文中所有词，非叶子节点都是一个逻辑二分类器
# 设每个分类器参数theta，分类器输入h，输出结果p为概率，p为左边，1-p为右边
#



from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# train数据里面每个sample是一个词条，后面跟着一个标签，0=文言文，1=现代文
# train的维度（5000，3），test（3385，2），train中有id，词条，标签，test中只有id和词条
train = pd.read_csv('/Users/lishuo/PycharmProjects/sofasofa/tutorial/word2vec/train.txt')
test = pd.read_csv('/Users/lishuo/PycharmProjects/sofasofa/tutorial/word2vec/test.txt')
texts = list(train['text'])+list(test['text'])

# 把每个字当作一个word，每个字都被表示为一个向量
# 训练的模型中，sentence是训练素材，为一个包含所有词条的列表；size指的是向量的长度，也就是每个词的维度；window是预测这个词之前之后的几个词。


# ndims 为设置每个字表示一个长度为50的向量
# ndims = 50
# model = Word2Vec(sentences=texts, size=ndims, window=5)

# model建立之后，可以通过model.wv['word']的格式来返还词向量
# print(model.wv['之'])

# 知道word vector 之后，可以通过 wv.most_similar,topn 来查看与之相近的几个词，词向量
# print(model.wv.most_similar('人',topn=10))

# 通过word2vec对文本进行可是化处理，现将每个字表示为长度2的向量，然后对每一句话中的所有字的向量求平均值，然后将每一句话表示为平面上一个点
newdim = 3
new_train = list(train['text'])

# model2 = Word2Vec(sentences=new_train,size=newdim,window=7)
# 定义total为一共多少词，vecs为所有词向量对应的矩阵

total  = len(new_train)
vecs = np.zeros([total,newdim])


# 对于训练集中所有的句子，一共5000个，对应它在列表中的index，进行如下操作：

# for index, sentence in enumerate(new_train):
#     counts,row =0,0
#     # 对于每一句话，遍历这句话中所有的词，尝试判断是否为空
#     for word in sentence:
#         try:
#             #如果改词不是空，则找到这个词对应的向量，计数+1
#             if word!=' ':
#                 row += model2.wv[word]
#                 counts +=1
#         except:
#             pass
#     if counts == 0:
#         print(sentence)
#     # 在完成一句的每个词的分析后，计算这句话中所有word向量的均值，也就是所有word的向量加起来，再除以总共多少个word
#     vecs[index,:] = row/counts

# # 画出图像，先定义图像大小
# plt.figure(figsize=(10,10))
# # 做color mapping，为一个list，然后list里面做映射，使用lambda x为红色，否则是蓝色，x值对应到train里面的标签
# colors = list(map(lambda x:'red' if x==1 else 'blue',train['y']))
# # colors 返回一个列表，里面全都是标签，为对应之后的结果
# print(colors)
# plt.scatter(vecs[:,0],vecs[:,1],c = colors,alpha=0.3,s=30,lw =1)
# plt.title('Word2Vec')
# plt.show()

# 建立文言文和现代文分类器

from sklearn.tree import DecisionTreeClassifier

num_data = len(train)+len(test)
n_train = len(train)
labeled_data = []
num_test  = len(test)

# 通过word2vec把所有库中的每个句子转化为维度指定维度的向量
sentence_dim = 10000
model1 = Word2Vec(sentences=texts,size=sentence_dim)

new_vec = np.zeros([num_data,sentence_dim])
for index, sentence in enumerate (texts):
    counts,row =0,0
    for word in sentence:
        try:
            if word != ' ':
                row += model1.wv[word]
                counts += 1
        except:
            pass
    if counts == 0:
        print(sentence)
    new_vec[index,:] = row / counts

# 生成分类器
tree_classifier = DecisionTreeClassifier(max_depth=4,random_state=100)

# 利用转化好的句子向量，取带标签的训练集中的句子向量和其对应标签用模型来拟合
tree_classifier.fit(new_vec[:n_train],train['y'])

submit = pd.read_csv('/Users/lishuo/PycharmProjects/sofasofa/tutorial/word2vec/sample_submit.csv')
# initiator = np.ones([num_test,1])*0.5
# submit['y'] = initiator
# submit.to_csv('/Users/lishuo/PycharmProjects/sofasofa/tutorial/word2vec/sample_submit.csv', index=False)

# 利用拟合好的模型来预测测试集的概率，由于是个二分类问题，我们只取第二列，也就是标签为1 的概率
submit['y'] = tree_classifier.predict_proba(new_vec[n_train:])[:,1]
submit.to_csv('my_prediction.csv', index=False)

test['pred'] = (submit['y'] > 0.5).astype(int)
print(test.head(20))