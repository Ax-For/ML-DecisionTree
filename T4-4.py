import numpy as np
import pandas as pd
from collections import Counter
from math import log
import copy

data_for_training = 'watermelon20_t.csv'
data_for_verification = 'watermelon20_v.csv'
raw_data = pd.read_csv(data_for_training,encoding='ansi')
X = raw_data.values[:,1:-1]
y = raw_data.values[:,-1]
X_names = np.array(raw_data.columns[1:-1])
y_name = raw_data.columns[-1]
result_map = {'是':'好瓜', '否':'坏瓜'}
veri_data = pd.read_csv(data_for_verification,encoding='ansi')
X_v = veri_data.values[:,1:-1]
y_v = veri_data.values[:,-1]
attributions = []
for i in range(len(X_names)):
    counter = Counter(X[:,i])
    s = []
    for k in counter.keys():
        s.append(k)
    attributions.append(s)


def gini(y):
    '''求解数据y的基尼指数'''
    counter = Counter(y)
    res = 1
    for v in counter.values():
        p = v / len(y)
        res -= p**2
    return res

def gini_index(y,y_s):
    '''求条件划分后的基尼指数'''
    g = 0
    for y_c in y_s:
        p = y_c.size / len(y)
        g += p * gini(y_c)
    return g

def split(X,y,d):
    '''对数据集在维度d上进行划分'''
    X_s, y_s = [], []
    for k in attributions[d]:
        X_s.append(X[X[:, d] == k])
        y_s.append(y[X[:, d] == k])
    return X_s, y_s

def try_split(X,y,D):
    """对X和y在属性集D中找出最优划分"""
    best_d = -1
    best_gi = float('inf')
    for d in D:
        X_s, y_s = split(X,y,d)
        gi = gini_index(y,y_s)
        if gi < best_gi:
            best_gi, best_d = gi, d
    return best_gi, best_d

class DeciTree:
    """决策树结构"""
    def __init__(self, kind, d, h=-1):
        """叶节点的高度设置为-1"""
        self.kind = kind
        self.d = d
        self.height = h
        self.children = {}

def generate_tree(X,y,D,height):
    """在高度为height的节点递归生成决策树"""
    kind = Counter(y).most_common(1)[0][0]
    #数据集D中样本类别相同
    if gini(y)==0:
        return DeciTree(kind,-1)
    #可用划分属性集为空
    if len(D)==0:
        return DeciTree(kind,-1)
    #样本在可用属性集上的取值相同
    else:
        flag = True
        for d in D:
            if len(Counter(X[:,d])) != 1:
                flag = False
                break
        if flag:
            return DeciTree(kind,-1)    
        else:
            gini_index, d = try_split(X,y,D)
            X_s, y_s = split(X,y,d)
            D_ = D.copy()
            D_.remove(d)
            DT = DeciTree(kind,d,height)
            for i in range(len(X_s)):
                #遇到空样本集
                if X_s[i].size == 0:
                    t = DeciTree(kind,-1)
                else:
                    t = generate_tree(X_s[i], y_s[i], D_,height+1)
                DT.children[attributions[d][i]] = t
            return DT

def decision_result(x,y,T):
    """返回验证样本(x,y)在决策树中的判定结果是否正确"""
    if T.d == -1:
        return T.kind == y
    else:
        return decision_result(x,y,T.children[x[T.d]])

def decision_precision(X,y,T):
    res = 0
    for x,y_ in zip(X,y):
        res += int(decision_result(x,y_,T))
    return res / len(y)

def print_tree(T, layer=0):
    """合理输出决策树"""
    if T.d == -1:
        print(' '*10*layer, 'leaf (result : {})'.format(result_map[T.kind]), sep='')
    else:
        print(' '*10*layer, 'branch 划分标准: {}'.format(X_names[T.d]),sep='')
        for d_value, tree in T.children.items():
            print(' '*10*(layer+1), '{}'.format(d_value), sep='')
            print_tree(tree, layer+1)

def pre_pruning(X, y, X_v, y_v, D):
    """生成决策树同时进行预剪枝"""
    kind = Counter(y).most_common(1)[0][0]
    # 数据集D中样本类别相同
    if gini(y) == 0:
        return DeciTree(kind, -1)
    # 可用划分属性集为空
    if len(D) == 0:
        return DeciTree(kind, -1)
    # 样本在可用属性集上的取值相同
    else:
        flag = True
        for d in D:
            if len(Counter(X[:, d])) != 1:
                flag = False
                break
        if flag:
            return DeciTree(kind, -1)
        else:
            gini_index, d = try_split(X, y, D)
            X_s, y_s = split(X, y, d)
            D_ = D.copy()
            D_.remove(d)
            T = DeciTree(kind, d)
            # 对可能进行的划分进行预剪枝
            # 设定变量分别表示不划分的准确数与划分后的准确度
            unsplit_precision, split_precision = 0, 0
            for i in range(len(y_v)):
                unsplit_precision += int(y_v[i] == T.kind)
                d_v = X_v[i, d]
                find = False
                for j in range(len(X_s)):
                    if X_s[j].size != 0 and X_s[j][0, d] == d_v:
                        find = True
                        break
                if find:
                    split_precision += int(y_v[i] == Counter(y_s[j]).most_common(1)[0][0])
                else:
                    split_precision += int(y_v[i] == T.kind)

            if split_precision > unsplit_precision:
                # 划分的性能更好
                for i in range(len(X_s)):
                    # 遇到空样本集
                    d_value = attributions[d][i]
                    if X_s[i].size == 0:
                        t = DeciTree(kind, -1)
                    else:

                        index = (X_v[:, d] == d_value)
                        X_vs, y_vs = X_v[index], y_v[index]
                        t = pre_pruning(X_s[i], y_s[i], X_vs, y_vs, D_)
                    T.children[d_value] = t
            else:
                T.d = -1
            return T

def get_highest_baunch(T):
    """找到T中的最高分支(height值最大)"""
    flag = True
    for t in T.children.values():
        if t.height!=-1:
            flag = False
            break
    #已经到达最高分支
    if flag:
        return T
    #还未到达最高分支
    else:
        h_height = -1
        h_baunch = T
        for child_tree in T.children.values():
            temp = get_highest_baunch(child_tree)
            if temp.height > h_height:
                h_height = temp.height
                h_baunch = temp
        return h_baunch


def post_pruning(X_v,y_v,T):
    """根据验证数据集对决策树T进行后剪枝"""
    if T.height == -1:
        T.height = 0
    else:
        highest_baunch = get_highest_baunch(T)
        while highest_baunch.height != 0:
            #剪枝前的精度
            before_pruning = decision_precision(X_v,y_v,T)
            #剪枝后的精度
            temp_d = highest_baunch.d
            highest_baunch.d = -1
            after_pruning = decision_precision(X_v,y_v,T)
            if after_pruning > before_pruning:
                pass
            else:
                highest_baunch.d = temp_d
            highest_baunch.height = -1
            highest_baunch = get_highest_baunch(T)

if __name__ == '__main__':
    D = set(range(X.shape[1]))
    T = generate_tree(X,y,D,0)
    a = decision_precision(X_v,y_v,T)
    print('*'*80)
    print('\n未剪枝：Accuracy: {:.3f}\n'.format(a))
    print_tree(T)
    T1 = pre_pruning(X,y,X_v,y_v,D)
    a1 = decision_precision(X_v,y_v,T1)
    print('\n','*'*80,sep='')
    print('\n预剪枝：Accuracy: {:.3f}\n'.format(a1))
    print_tree(T1)
    T2 = T
    post_pruning(X_v,y_v,T2)
    a2 = decision_precision(X_v,y_v,T2)
    print('\n','*'*80,sep='')
    print('\n后剪枝：Accuracy: {:.3f}\n'.format(a2))
    print_tree(T2)
    print('\n','*'*80,sep='')