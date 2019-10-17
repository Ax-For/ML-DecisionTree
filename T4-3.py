
import numpy as np
import pandas as pd
from collections import Counter
from math import log
data_file_name = "watermelon30.csv"
df = pd.read_csv(data_file_name,encoding='ansi')
X = df.values[:,1:-1]
y = df.values[:,-1]
X_names = np.array(df.columns[1:-1])
y_name = df.columns[-1]
attributions = []
for i in range(len(X_names)):
    if X_names[i] not in ('密度','含糖率'):
        counter = Counter(X[:,i])
        s = []
        for k in counter.keys():
            s.append(k)
        attributions.append(s)
    else:
        attributions.append([])

def entropy(y):
    """返回y的信息熵"""
    counter = Counter(y)
    res = 0
    for i in counter.values():
        p = i / len(y)
        res += -p * log(p)
    return res

def split(X, y, d, s=float('-inf')):
    """对数据针对d维进行划分"""
    if X_names[d] not in ('密度','含糖率'):
        '''对离散值的处理'''
        X_s, y_s = [], []
        for k in attributions[d]:
            X_s.append(X[X[:, d] == k])
            y_s.append(y[X[:, d] == k])
        return X_s, y_s
    else:
        '''对连续值的处理，进行二分类'''
        X_s, y_s = [], []
        X_s.append(X[ X[:,d]<=s ])
        X_s.append(X[ X[:,d]>s ])
        y_s.append(y[ X[:,d]<=s ])
        y_s.append(y[ X[:,d]>s ])
        return X_s, y_s

def try_split(X,y,D):
    '''对数据集X,y和属性集D,求可能划分'''
    best_e = float('inf')   #熵
    best_d = -1             #切分维度
    best_s = float('-inf')  #切分值
    for d in D:
        if X_names[d] not in ('密度','含糖率'):
            X_s, y_s = split(X, y, d)
            e = 0
            for item in y_s:
                e += item.size / len(y) * entropy(item)
            if e < best_e:
                best_d, best_e = d, e
        else:
            index_sorted = np.argsort(X[:,d])
            for i in range(1, len(X)):
                if X[index_sorted[i], d] != X[index_sorted[i-1], d]:
                    s = (X[index_sorted[i], d] + X[index_sorted[i-1], d]) / 2
                    X_s, y_s = split(X,y,d,s)
                    e = y_s[0].size / len(y) * entropy(y_s[0]) + y_s[1].size / len(y) * entropy(y_s[1])
                    if e < best_e:
                        best_d, best_e, best_s = d, e, s
    return best_d, best_e, best_s

class DTree:
    """构建兼容连续值与离散值的决策树"""
    def __init__(self, kind, dim, s=float('-inf')):
        self.kind = kind
        self.dim = dim
        self.s = s
        self.children = {}

def generate_Tree(X,y,D):
    """根据输入输出值以及可划分属性进行决策树生成"""
    kind = Counter(y).most_common(1)[0][0]
    #数据集D中的样本为同一类
    if entropy(y) == 0:
        return DTree(kind, -1)
    #数据集中的离散可用划分属性已使用完
    if len(D) == 2:
        return DTree(kind, -1)
    else:
        flag = True
        for d in D - {6,7}:
            if len(Counter(X[:,d])) != 1:
                flag = False
                break
        if flag:
            return DTree(kind)
        else:
            d, e, s = try_split(X,y,D)
            X_s, y_s = split(X,y,d,s)
            if X_names[d] not in ('密度','含糖率'):
                D_ = D.copy()
                D_.remove(d)
                DT = DTree(kind,d)
                for i in range(len(X_s)):
                    #遇到空样本集
                    if X_s[i].size == 0:
                        t = DTree(kind,-1)
                    else:
                        t = generate_Tree(X_s[i], y_s[i], D_)
                    DT.children[attributions[d][i]] = t
                return DT
            else:
                DT = DTree(kind, d, s)
                t0 = generate_Tree(X_s[0], y_s[0], D)
                DT.children['<='+str(s)] = t0
                t1 = generate_Tree(X_s[1], y_s[1], D)
                DT.children['>'+str(s)] = t1
                return DT

def print_tree(T, layer=0):
    """合理输出决策树"""
    target_map = {'是':'好瓜', '否':'坏瓜'}
    if T.dim == -1:
        print(' '*10*layer, 'leaf (result :{})'.format(target_map[T.kind]), sep='')
    else:
        print(' '*10*layer, 'branch  划分标准: {}'.format(X_names[T.dim]),sep='')
        for d_value, tree in T.children.items():
            print(' '*10*(layer+1), '{}'.format(d_value), sep='')
            print_tree(tree, layer+1)

if __name__ == '__main__':
    D = set(range(X.shape[1]))
    T = generate_Tree(X,y,D)
    print_tree(T)
