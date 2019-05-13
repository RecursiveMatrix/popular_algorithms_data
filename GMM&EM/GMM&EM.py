# GMM: 高斯混合模型，聚类模型，考虑均值和协方差，同时假设数据是由几个不同的高斯随机变量分布组合而成
# 此模型下，目的是估计每一个高斯分布的均值和方差，即对一个n个样本，k类的数据，需要找到k个平均值和方差来最大化此条件下的似然函数
# 间接引入隐变量矩阵W, 此时的W是一个n*k的矩阵，矩阵中的每个元素体现了这一元素所在行，即第n个样本是第k个类的概率
# EM最大期望算法，即希望找到一个最佳的W，从而得到GMM的模型参数：用期望跟新W，跟新所有高斯变量参数


# step 1 生成数据
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')

from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal

# 定义五个二维高斯分布参数如下：样本个数，平均值，方差
num1,av1,var1 = 400,[0.5,0.5],[1,3]
num2,av2,var2 = 600,[5.6,3.0],[2,2]
num3,av3,var3 = 1000,[2,10],[6,3]
num4,av4,var4 = 1400,[9,1],[2,8]
num5,av5,var5 = 2000,[8,8],[8,2]


# 通过multivarite normal以及参数生成样本
X1 = np.random.multivariate_normal(av1,np.diag(av1),num1)
X2 = np.random.multivariate_normal(av2,np.diag(av2),num2)
X3 = np.random.multivariate_normal(av3,np.diag(av3),num3)
X4 = np.random.multivariate_normal(av4,np.diag(av4),num4)
X5 = np.random.multivariate_normal(av5,np.diag(av5),num5)
X = np.vstack((X1,X2,X3,X4,X5))

plt.figure(figsize=(10,8))
plt.scatter(X1[:,0],X1[:,1],s=6)
plt.scatter(X2[:,0],X2[:,1],s=6)
plt.scatter(X3[:,0],X3[:,1],s=6)
plt.scatter(X4[:,0],X4[:,1],s=6)
plt.scatter(X5[:,0],X5[:,1],s=6)

plt.show()

# step 2 初始化所需变量

# 目标分类个数
k_cluster = 5
# 所有样本合并在一起
n_sample = len(X)
# 初始化平均数和方差
AV = [[1,1],[5,4],[1,8],[10,3],[9,10]]
VAR = [[1,1],[1,1],[1,1],[1,1],[1,1]]
# 初始化每个类的可能性，这个可能性最后体现了几个类数量所占的比例
prob_cluster = [1/n_sample]*k_cluster
# 初始化n*k的矩阵
W = np.ones((n_sample,k_cluster))/k_cluster
# 按照每一列求和，再除以所有元素的和，以此来跟新不同类的所占比例
prob_cluster = W.sum(axis=0)/W.sum()

# 更新W矩阵
# 所需参数，数据集，初始化的平均值和方差，每个类所占的比例

def update_W(X,AV,VAR,prob_cluster):

    n_sample,k_cluster = len(X),len(prob_cluster)
    # 生成n*k的矩阵
    pdfs = np.zeros(((n_sample,k_cluster)))
    # 针对每个类别，循环所有样本，通过每一类所占比例和所属这一类的概率相乘
    for i in range(k_cluster):
        pdfs[:,i] = prob_cluster[i]*multivariate_normal.pdf(X,AV[i],np.diag(VAR[i]))
    # 最后将W重新scale：对于每个样本，横向求和，reshape成一列，然后通过原数据reshape
    W = pdfs/pdfs.sum(axis=1).reshape(-1,1)

    return W    # n*k

# 通过权重矩阵来更新每一类的分布
def update_prob_cluster(W):
    # 把W矩阵按照列求和，即求出每个样本是某一类概率的所有的和，再除以总矩阵的和
    prob_cluster = W.sum(axis=0)/W.sum()

    return prob_cluster     # k*1

# step 3 计算似然函数

def logLH(X,prob_cluster,AV,VAR):

    n_sample,k_cluster = len(X),len(prob_cluster)
    pdfs = np.zeros(((n_sample,k_cluster)))
    for i in range(k_cluster):
        pdfs[:,i] = prob_cluster[i]*multivariate_normal.pdf(X,AV[i],np.diag(VAR[i]))

    # log似然函数和权重矩阵类似，只不过最后返回的是将pdfs按行求出和，再算log，再求出平均值
    # log函数为一个衡量标准，旨在最大化所有样本的似然程度
    return np.mean(np.log(pdfs.sum(axis=1)))        # scaler



# step 4 更新 AV 和 VAR
def update_AV(X,W):

    AV = np.zeros((k_cluster,2))
    for i in range(k_cluster):
        AV[i] = np.average(X,axis=0,weights=W[:,i])
    return AV

def update_VAR(X,AV,W):

    VAR = np.zeros((k_cluster,2))
    for i in range(k_cluster):
        VAR[i] = np.average((X-AV[i])**2,axis=0,weights=W[:,i])
    return VAR

# step 5 画出图像

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def plot_cluster(X, AV, VAR, AV_real=None, VAR_real=None):
    # 定义颜色，与类别数一致
    cmap = get_cmap(k_cluster)

    plt.figure(figsize=(10, 8))
    plt.axis([-5, 23, -5, 23])
    plt.scatter(X[:, 0], X[:, 1], s=7)

    # 画出模型生成的类范围
    ax = plt.gca()
    for m in range(k_cluster):
        plot_args = {'fc': 'None', 'lw': 2, 'edgecolor': cmap(m), 'ls': ':'}
        ellipse = Ellipse(AV[m], 2 * VAR[m][0], 2 * VAR[m][1], **plot_args)
        print( VAR[m][0], VAR[m][1])
        ax.add_patch(ellipse)

    # 画出实际类的范围
    if AV_real and VAR_real:
        for m in range(k_cluster):
            plot_args = {'fc': 'None', 'lw': 2, 'edgecolor': cmap(m), 'alpha': 0.5}
            ellipse = Ellipse(AV_real[m], 2 * VAR_real[m][0], 2 * VAR_real[m][1], **plot_args)
            print(VAR_real[m][0], VAR_real[m][1])

            ax.add_patch(ellipse)
    plt.show()



if __name__ == '__main__':

    logLh = []
    for i in range(5):
        plot_cluster(X, AV, VAR, [av1, av2, av3, av4, av5], [var1, var2, var3, var4, var5])
        logLh.append(logLH(X, prob_cluster, AV, VAR))
        W = update_W(X, AV, VAR, prob_cluster)
        Pi = update_prob_cluster(W)
        AV = update_AV(X, W)
        print('log-likehood:%.3f'%logLh[-1])
        VAR = update_VAR(X, AV, W)