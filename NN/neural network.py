import numpy as np

input_data = np.array([[0, 0, 1],
                       [0, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])

output_labels = np.array([[0],
                          [1],
                          [1],
                          [0]])

# 1 定义输入输出，来自数据

def activate(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))


# 2 根据输入数据特征值a 和输出列数b 构建一个初始weight 矩阵 a by b
# weight0 是input与layer1 之间 4by3 * 4by4
weight_0 = 2*np.random.random((3,4)) - 1
# weight1 是layer1与output 之间 4by4 *4by1
weight_1 = 2*np.random.random((4,1)) - 1

for j in range(60000):

    # Forward propagate through layers 0, 1, and 2
    layer0 = input_data

    # 3 将矩阵数据乘积经过激活函数mapping
    layer1 = activate(np.dot(layer0, weight_0))# 4by4
    layer2 = activate(np.dot(layer1, weight_1))# 4by1

    # 4 计算误差 注意是从后往前做差
    layer2_error = output_labels - layer2

    if (j % 10000) == 0:
        print("Error:" + str(np.mean(np.abs(layer2_error))))

    # Use it to compute the gradient
    # 5 用这一层得到的误差，套入激活函数的导数，算出梯度值
    layer2_gradient = layer2_error * activate(layer2, deriv=True) # 4by1

    # calculate error for layer 1
    # 前一层的误差可以通过上一层得到的梯度值与前一层的乘积得到，再与上一层的weight的转至矩阵求乘积
    layer1_error = layer2_gradient.dot(weight_1.T)# 4by4

    # Use it to compute its gradient
    layer1_gradient = layer1_error * activate(layer1, deriv=True) # 4by4

    # update the weights using the gradients
    weight_1 += layer1.T.dot(layer2_gradient)#4by1
    weight_0 += layer0.T.dot(layer1_gradient)#4by4

