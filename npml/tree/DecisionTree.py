from ..base import ClassifierMixin


def divide_set(rows, column, value):
    """在列column上对数据集进行拆分，能够处理数值型数据或名词型数据"""
    # 如果数值型 按大小拆分 如果非字符型 按值拆分
    def split_function(row):
        return row[column] >= value if isinstance(value, int) or isinstance(value, float) else row[column] == value
    # 将数据集拆分成两个集合，并返回
    set1 = [row for row in rows if split_function(row)]
    set2 = [row for row in rows if not split_function(row)]
    return set1, set2


class DecisionTree(ClassifierMixin):
    def __init__(self):
        super(DecisionTree, self).__init__()

    def load_data(self):
        """加载数据"""
        my_data = [['slashdot', 'USA', 'yes', 18, 'None'],
                   ['google', 'France', 'yes', 23, 'Premium'],
                   ['digg', 'USA', 'yes', 24, 'Basic'],
                   ['kiwitobes', 'France', 'yes', 23, 'Basic'],
                   ['google', 'UK', 'no', 21, 'Premium'],
                   ['(direct)', 'New Zealand', 'no', 12, 'None'],
                   ['(direct)', 'UK', 'no', 21, 'Basic'],
                   ['google', 'USA', 'no', 24, 'Premium'],
                   ['slashdot', 'France', 'yes', 19, 'None'],
                   ['digg', 'USA', 'no', 18, 'None'],
                   ['google', 'UK', 'no', 18, 'None'],
                   ['kiwitobes', 'UK', 'no', 19, 'None'],
                   ['digg', 'New Zealand', 'yes', 12, 'Basic'],
                   ['slashdot', 'UK', 'no', 21, 'None'],
                   ['google', 'UK', 'yes', 18, 'Basic'],
                   ['kiwitobes', 'France', 'yes', 19, 'Basic']]
        self.data = my_data


class Node(object):
    """决策树上的节点"""

    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
        """每个节点有5个变量"""
        self.col = col  # 待检验的判断条件所对应的列的索引值
        self.value = value  # 为了使结果为true，当前列必须匹配的值
        self.results = results  # 针对于当前分支的结果，是一个字段，除叶节点外，其他节点上该值都为None
        self.tb = tb  # 结果为true的子节点
        self.fb = fb  # 结果为false的子节点


if __name__ == '__main__':
    tree = DecisionTree()
    tree.load_data()
    print(divide_set(tree.data, 2, 'yes'))
