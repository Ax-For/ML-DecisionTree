import numpy as np
import pandas as pd
from collections import Counter
from math import log

data_file_name = 'T5-1.csv'
df = pd.read_csv(data_file_name,encoding='ansi')
X = df.values[:,1:-1]
y = df.values[:,-1]
X_names = np.array(df.columns[1:-1])
y_name = df.columns[-1]
attributions = []
for i in range(len(X_names)):
	counter = Counter(X[:,i])
	s = []
	for k in counter.keys():
		s.append(k)
	attributions.append(s)

# 关于决策树的构建
class DecisionTree:
    def __init__(self, kind, dim=-1):
        """决策树类，属性为决策结果（kind）和切分维度（dim）"""
        self.kind = kind
        self.dim = dim
        self.children = {}


def entropy(y):
    """计算信息熵"""
    counter = Counter(y)
    res = 0
    for num in counter.values():
        p = num / len(y)
        res += -p * log(p)
    return res


def gain_ratio(y, y_s):
    """对于针对某个特征进行划分后的数据集求信息增益比"""
    condition_entropy = 0
    for yc in y_s:
        if yc.size!=0:
        	p = yc.size / len(y)
        	condition_entropy += p * entropy(yc)
    return 1 - condition_entropy / entropy(y)


def split(X, y, d):
    """根据在d维的取值不同进行划分"""
    X_s, y_s = [], []
    for k in attributions[d]:
        X_s.append(X[X[:, d] == k])
        y_s.append(y[X[:, d] == k])
    return X_s, y_s


def try_split(X, y, D):
    """尝试进行划分，需要返回最佳划分维度d"""
    best_gain_r = float('-inf')
    best_d = -1
    for d in D:
        X_s, y_s = split(X, y, d)
        gain_r = gain_ratio(y, y_s)
        if gain_r > best_gain_r:
            best_gain_r, best_d = gain_r, d
    return best_gain_r, best_d


def generate_DT(X, y, D):
    """传入输入数据，输出数据与可用属性集合，递归进行决策树生成"""
    kind = Counter(y).most_common(1)[0][0]
    #样本属于同一类别
    if entropy(y) == 0:
        return DecisionTree(kind)
    #属性集已空
    if len(D) == 0:
        return DecisionTree(kind)
    #所有样本在已知属性集上取值相同
    else:
        flag = True
        for d in D:
            if len(Counter(X[:,d])) != 1:
                flag = False
                break
        if flag:
            return DecisionTree(kind)
        else:
            #正常进行划分
            gain_r, d = try_split(X, y, D)
            X_s, y_s = split(X, y, d)
            D_ = D.copy()
            D_.remove(d)
            DT = DecisionTree(kind, d)
            for i in range(len(X_s)):
            	#遇到空样本集
                if X_s[i].size == 0:
                    t = DecisionTree(kind)
                else:
                    t = generate_DT(X_s[i], y_s[i], D_)
                DT.children[attributions[d][i]] = t
    return DT


def print_tree(T, layer=0):
    """合理输出决策树"""
    if T.dim == -1:
        print(' '*8*layer, 'leaf (result :{})'.format(T.kind), sep='')
    else:
        print(' '*8*layer, 'branch  划分标准: {}'.format(X_names[T.dim]),sep='')
        for d_value, tree in T.children.items():
            print(' '*8*(layer+1), '{}'.format(d_value), sep='')
            print_tree(tree, layer+1)


if __name__ == '__main__':
    D = set(range(X.shape[1]))
    DTree = generate_DT(X, y, D)
    print_tree(DTree)
