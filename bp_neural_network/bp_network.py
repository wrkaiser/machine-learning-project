from DL import bp_neural_network


class Network(object):

    # 初始化一个全连接神经网络
    # layers: 二维数组，描述神经网络每层节点数
    def __init__(self, layers):
        self.connections = bp_neural_network.connections.Connection()  # 存储所有链接的池子
        self.layers = []
        layer_count = len(layers)
        node_count = 0
        for i in range(layer_count):
            self.layers.append(bp_neural_network.layer.Layer(i, layers[i]))
        for layer in range(layer_count - 1):  # 全连接
            connections = [bp_neural_network.connections.Connection(upstream_node, downstream_node)
                           for upstream_node in self.layers[layer].nodes
                           for downstream_node in self.layers[layer + 1].nodes[:-1]]
            for conn in connections:
                self.connections.add_connection(conn)
                conn.downstream_node.append_upstream_connection(conn)
                conn.upstream_node.append_downstream_connection(conn)

    #  训练神经网络
    #  labels: 数组，训练样本标签。每个元素是一个样本的标签。
    #  data_set: 二维数组，训练样本特征。每个元素是一个样本的特征。
    def train(self, labels, data_set, rate, iteration):
        for i in range(iteration):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d], data_set[d], rate)

    # 内部函数，用一个样本训练网络
    def train_one_sample(self, label, sample, rate):
        self.predict(sample)
        self.calc_delta(label)
        self.update_weight(rate)

    # 内部函数，计算每个节点的delta
    def calc_delta(self, label):
        output_nodes = self.layers[-1].nodes
        for i in range(len(label)):
            output_nodes[i].calc_output_layer_delta(label[i])
        for layer in self.layers[-2::-1]:
            for node in layer.nodes:
                node.calc_hidden_layer_delta()

    #  内部函数，更新每个连接权重
    def update_weight(self, rate):
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.update_weight(rate)

    # 内部函数，计算每个连接的梯度
    def calc_gradient(self):
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.calc_gradient()

    # 获得网络在一个样本下，每个连接上的梯度
    # label: 样本标签
    # sample: 样本输入
    def get_gradient(self, label, sample):
        self.predict(sample)
        self.calc_delta(label)
        self.calc_gradient()

    # 根据输入的样本预测输出值
    # sample: 数组，样本的特征，也就是网络的输入向量
    def predict(self, sample):
        self.layers[0].set_output(sample)
        for i in range(1, len(self.layers)):
            self.layers[i].calc_output()
        return map(lambda node: node.output, self.layers[-1].nodes[:-1])

    # 打印网络信息
    def dump(self):
        for layer in self.layers:
            layer.dump()