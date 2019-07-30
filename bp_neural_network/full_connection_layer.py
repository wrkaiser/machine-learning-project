import numpy as np


# 全连接层实现类
class FullConnectedLayer(object):

    def __init__(self, input_size, output_size,
                 activator):
        '''
        构造函数
        input_size: 本层输入向量的维度
        output_size: 本层输出向量的维度
        activator: 激活函数
        '''
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator
        # 权重数组W
        self.W = np.random.uniform(-0.1, 0.1, (output_size, input_size))
        # 偏置项b
        self.b = [0 for _ in range(0, output_size)]
        # 输出向量
        self.output = np.zeros((output_size, 1))

    # 前向计算  input_array: 输入向量
    def forward(self, input_array):
        # 式2
        self.input = input_array
        temp = np.dot(self.W, input_array)   # 并非矩阵乘，向量乘
        rs = temp + self.b   # 得到输出节点的输入
        self.output = self.activator.forward(
            rs)  # 输出节点的输出，作为下一层的输入

    # 反向传播 从输出层的前一层开始调用
    def backward(self, delta_array):
        '''
        反向计算W和b的梯度
        delta_array: 从上一层传递过来的误差项
        '''
        # 式8
        # temp_delta_array = np.array(delta_array).reshape(-1, 1)

        temp = np.dot(self.W.T, delta_array.tolist())
        print(temp)
        self.delta = self.activator.backward(self.input) * temp  # 计算非输出层的误差项
        self.W_grad = np.dot(delta_array.tolist(), self.output)
        self.b_grad = delta_array

    def update(self, learning_rate):
        '''
        使用梯度下降算法更新权重
        '''
        self.W += learning_rate * self.W_grad
        self.b += learning_rate * self.b_grad