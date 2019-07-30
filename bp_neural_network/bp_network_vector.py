import numpy as np
from DL.bp_neural_network import full_connection_layer


# Sigmoid激活函数类
class SigmoidActivator(object):

    def forward(self, weighted_input):
        print("!!!!" + str(weighted_input.shape) + "!!!!")
        result = 1.0 / (1.0 + np.exp(-weighted_input))
        print(result.shape)
        return result

    def backward(self, output):
        return np.array(output) * (1 - np.array(output))


# 神经网络类
class Network(object):

    def __init__(self, layers):
        '''
        构造函数
        '''
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(
                full_connection_layer.FullConnectedLayer(
                    layers[i], layers[i+1],
                    SigmoidActivator()
                )
            )

    def predict(self, sample):
        '''
        使用神经网络实现预测
        sample: 输入样本
        '''
        output = sample
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output

    def train(self, labels, data_set, rate, epoch):
        '''
        训练函数
        labels: 样本标签
        data_set: 输入样本
        rate: 学习速率
        epoch: 训练轮数
        '''
        for i in range(epoch):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d],
                    data_set[d], rate)

    def train_one_sample(self, label, sample, rate):
        # sample 和 label 均为行list
        self.predict(sample)
        self.calc_gradient(label)
        self.update_weight(rate)

    def calc_gradient(self, label):
        #print(label)
        #print(self.layers[-1].output)
        delta = self.layers[-1].activator.backward(self.layers[-1].output) * (label - self.layers[-1].output)  # 输出层的误差项
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta
        return delta

    def update_weight(self, rate):
        for layer in self.layers:
            layer.update(rate)