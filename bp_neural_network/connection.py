import random


class Connection(object):

    # 初始化连接，权重初始化为是一个很小的随机数
    # upstream_node: 连接的上游节点
    # downstream_node: 连接的下游节点
    def __init__(self, upstream_node, downstream_node):
        self.upstream_node = upstream_node
        self.downstream_node = downstream_node
        self.weight = random.uniform(-0.1, 0.1)
        self.gradient = 0.0

    #  计算梯度
    def calc_gradient(self):
        self.gradient = self.downstream_node.delta * self.upstream_node.output

    #  获取梯度
    def get_gradient(self):
        return self.gradient

    # 根据梯度下降算法更新权重
    def update_weight(self, rate):
        self.calc_gradient()
        self.weight += rate * self.gradient

    # 打印连接信息
    def __str__(self):
        return '(%u-%u) -> (%u-%u) = %f' % (
            self.upstream_node.layer_index,
            self.upstream_node.node_index,
            self.downstream_node.layer_index,
            self.downstream_node.node_index,
            self.weight)