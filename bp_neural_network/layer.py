from DL.bp_neural_network import node
from DL.bp_neural_network import const_node


class Layer(object):

    # 初始化一层
    # layer_index: 层编号
    # node_count: 层所包含的节点个数
    def __init__(self, layer_index, node_count):
        self.layer_index = layer_index
        self.nodes = []
        for i in range(node_count):
            self.nodes.append(node.Node(layer_index, i))
        self.nodes.append(const_node.ConstNode(layer_index, node_count))

    # 设置层的输出。当层是输入层时会用到。
    def set_output(self, data):
        for i in range(len(data)):
            self.nodes[i].set_output(data[i])

    # 计算层的输出向量
    def calc_output(self):
        for node in self.nodes[:-1]:
            node.calc_output()

    #  打印层的信息
    def dump(self):
        for node in self.nodes:
            print(node)