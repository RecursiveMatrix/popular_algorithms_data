# 条件概率bayes：
# 在B发生情况下A发生的概率 = AB一起发生的概率/B发生的概率
# 朴素贝叶斯是一个分类算法，假设所以特征相互独立
# 例如，特征x1，x2，标签y；在已知x1，x2的情况下，预测标签为1的概率 = 当前x1，x2情况下y是1的概率 / x1，x2是当前情况下的概率
# 某种情况下输出概率预测 = 先验概率（已知该输出的概率）*似然（这个输出时，特征的情况）/证据（当特征是当前情况的概率）
# 由于假设独立，似然可以写成当前输出结果下，所有特征出现这个情况的概率的乘积


# 问题： 求某个名字人的性别
# P(性别|名字） = P(性别）*P（名字|性别）/ P(名字）

import numpy as np
import pandas as pd
import math
from collections import defaultdict

train = pd.read_csv('train.txt')
test = pd.read_csv('test.txt')


# check if the distribution is biased or not
name_female = train[train['gender']==0]
name_male = train[train['gender']==1]
total = {'f':len(name_female), 'm':len(name_male)}


# 在所有名字里面，求某个字出现的概率，相当于P（名字|性别） = P(名字|女生)和P（名字|男生）
freq_list_female = defaultdict(int)
for name in name_female['name']:
    for char in name:
        freq_list_female[char] += 1./total['f']



freq_list_male = defaultdict(int)
for name in name_male['name']:
    for char in name:
        freq_list_male[char] += 1./total['m']

# 通过这个结果可以看出，当已知男生或者女生的前提时，某一个字出现的概率是多少

# 由于预测集中可能有些字不出现在训练集里，需要对频率进行拉普拉斯平滑
# Laplace平滑： 例子： '我爱机器学习' '我'的频率为1/7 ， '学'的频率为2/7
# 平滑后的频率 g（x） = （单词x出现的次数 + Laplace平滑系数）/（句子长度 + Laplace平滑系数*句子中不同词汇的个数）
# 如果平滑系数越大，结果越接近于均匀分布
# 这个方法可以解决一些0概率问题，即某一个样本即使不出现，也不能认为它的概率是0

def LaplaceSmooth(char,freq_list,total,alpha = 1.0):

    count = freq_list[char] * total
    distinct_chars = len(freq_list)
    freq_laplace = (count+alpha)/(total + distinct_chars*alpha)

    return freq_laplace


# 对于一个分类问题，只需要得到标签即可，即通过以下预测
# y_pred = argmax(P(Y=y)*P(X1|Y=y)*...P(Xi|Y=y))

# 鉴于在性别预测中，大量特征都为0， P(Xi)数值比较小，防止浮点误差，两边取对数
# 浮点误差：四舍五入造成的误差
# Ypred = logP(Y=y) + sum(logP(Xi=0|Y=y) + logP(Xi=1|Y=y) - logP(Xi=0|Y=y)
# 例： 如果一个人有两个名字，假设X1=1,X4=1，其余都是0
# 那么： Ypred = logP(Y=y) + sum(logP(Xi=0|Y=y) + logP(X1=1|Y=y) - logP(X1=0|Y=y) + logP(X4=1|Y=y) - logP(X4=0|Y=y)
# 也就是需要算 先验概率logP(Y=y)，
# 似然概率：由于大部分都是0，所以可以简化为：先算当前标签下，概率都是0的概率之和；加上某个位置在当前标签下有特征的概率-这个位置当前标签下没特征的概率

# 建立一个base字典，计算 logP(Y=y) + sum(logP(Xi=0|Y=y)
# 由于是一个二分类问题，每个性别只需计算一遍

# 在训练集中，由于只有male被标记为1，所以可以通过 train['gender'].mean() 来计算男性名字所占比例，用1-这个比例就是女性名字的比例
logP_f = math.log(1-train['gender'].mean())
logP_m = math.log(train['gender'].mean())
# 同理可以按照下式子求
# total['f']/(total['f']+total['m'])
# total['m']/(total['f']+total['m'])

# 在刚才算出先验概率部分的基础上加上： 当前标签下所有特征为0的概率
# 刚才计算的freq list分别算出某个字出现在男女标签下的概率，所以1-对应的概率就是特征为0 的概率
# 同时经过一个for循环写在一个list里，再用sum函数求和
logP_f += sum([math.log(1-freq_list_female[char]) for char in freq_list_female])
logP_m += sum([math.log(1-freq_list_male[char]) for char in freq_list_male])

base = {'f':logP_f,'m':logP_m}
print(base['f'],base['m'])

# 计算 logP(Xi=1|Y=y) - logP(Xi=0|Y=y)
# 调用该函数，需要已知那个词，也就是Xi，那个list，也就是已知的标签，同时需要知道total，也就是所有先验概率，这里用频数表述--用来平滑

def LogProb(char,freq_list,total):

    freq_smooth = LaplaceSmooth(char,freq_list,total)
    return math.log(freq_smooth) - math.log(1-freq_smooth)



def Prob(name,bases,totals,freq_list_male,freq_list_female):

    logP_m = bases['m'] # 概率相同部分
    logP_f = bases['f']
    for char in name:
        logP_m += LogProb(char,freq_list_male,totals['m'])  # 有那个字的概率
        logP_f += LogProb(char,freq_list_female,totals['f'])

    return {'male':logP_m,'female':logP_f}  # 返回一个字典，分别是两个性别的可能性

def gender_identifier(log_P):   # 哪个值大就拿谁作为标签

    return log_P['male']>log_P['female']    # 返回True or False


result = []

for name in test['name']:
    log_P = Prob(name,base,total,freq_list_male,freq_list_female)
    gender = gender_identifier(log_P)
    result.append(int(gender))  # True False经过int转型变为1，0




submit = pd.read_csv('/Users/lishuo/PycharmProjects/sofasofa/tutorial/bayes/sample_submit.csv')

submit['gender'] = result

submit.to_csv('my_NB_prediction.csv', index=False)

test['pred'] = result
print(test.head(20))

# 总结：
# 1 先把训练集按照性别分成两个组，计算每个组性别（标签）个数，通过大数定律，可得到先验概率，也就是是男是女的概率
# 2 根据两个组分别求似然概率，在性别男的标签下计算每个字出现的概率，在性别女的标签下计算每个字出现的概率
# 3 把似然概率结果进行Laplace处理，目的是防止测试集出现训练集里没有的字，平滑过程需要：当前字出现次数，句子长度，参数，句子中包含的不同字
# 4 根据贝叶斯公式：先算出这个性别（标签）下的概率，取log，再加上所有字都没有出现的概率（即 1-出现字的概率，已经通过Laplace得到），求和，再取对数
# 5 由得出的概率判断标签类别
