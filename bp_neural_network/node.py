from functools import reduce
import numpy as np


# 节点类，负责记录和维护节点自身信息以及与这个节点相关的上下游连接，实现输出值和误差项的计算。
class Node(object):

    # 构造节点对象。
    # layer_index: 节点所属的层的编号
    # node_index: 节点的编号
    def __init__(self, layer_index, node_index):
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.upstream = []
        self.output = 0
        self.delta = 0

    # 计算sigmod
    @staticmethod
    def sigmoid(input):
        return 1. / (1. + np.exp(-input))

    # 设置节点的输出值。如果节点属于输入层会用到这个函数。
    def set_output(self, output):
        self.output = output

    # 添加该node的一个下游节点
    def append_downstream_connection(self, conn):
        self.downstream.append(conn)

    # 添加一个到上游节点的连接
    def append_upstream_connection(self, conn):
        self.upstream.append(conn)

    # 根据式1计算节点的输出
    def calc_output(self):
        output = reduce(lambda ret, conn: ret + conn.upstream_node.output * conn.weight, self.upstream, 0)
        self.output = Node.sigmoid(output)

    # 节点属于隐藏层时，根据式4计算delta
    def calc_hidden_layer_delta(self):
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
            self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    # 节点属于输出层时，根据式3计算delta
    def calc_output_layer_delta(self, label):
        self.delta = self.output * (1 - self.output) * (label - self.output)

    # 打印节点的信息
    def __str__(self):
        node_str = '%u-%u: output: %f delta: %f' % (self.layer_index, self.node_index, self.output, self.delta)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        upstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.upstream, '')
        return node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str