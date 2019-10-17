import numpy as np
from math import log
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
raw_X = iris.data
raw_y = iris.target
X_names = iris.feature_names
res_names = iris.target_names
#划分验证集与测试集
X, X_v, y, y_v = train_test_split(raw_X,raw_y,test_size=0.1,random_state=24)

#对数据集iris执行（信息熵）分类任务算法(输入均为连续值)(包括预剪枝)
def entropy(y):
    """返回y的信息熵"""
    counter = Counter(y)
    res = 0
    for i in counter.values():
        p = i / len(y)
        res += -p * log(p)
    return res

def split(X,y,d,s):
    """对数据集(X,y)在d维根据切分值s进行划分"""
    X_s,y_s = [],[]
    X_s.append(X[ X[:,d]<=s ])
    X_s.append(X[ X[:,d]>s ])
    y_s.append(y[ X[:,d]<=s ])
    y_s.append(y[ X[:,d]>s ])
    return X_s, y_s 

def try_split(X,y,D):
    """找出最佳切分维度与最佳切分变量"""
    best_e = float('inf')   #熵
    best_d = -1             #切分维度
    best_s = float('-inf')  #切分值
    for d in D:
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
    def __init__(self, kind, d=-1, height=-1,s=float('-inf')):
        self.kind = kind
        self.d = d
        self.height = height
        self.s = s
        self.children = {}

def generate_tree(X,y,D,height):
    """根据数据集(X,y)生成训练模型决策树"""
    kind = Counter(y).most_common(1)[0][0]
    #数据集D中的样本为同一类
    if entropy(y) == 0:
        return DTree(kind, -1)
    else:
        flag = True
        for d in D:
            if len(Counter(X[:,d])) != 1:
                flag = False
                break
        #数据集在给定属性集合上取值相同
        if flag:
            return DTree(kind)
        else:
            d, e, s = try_split(X,y,D)
            X_s, y_s = split(X,y,d,s)
            DT = DTree(kind, d, height, s)
            t0 = generate_tree(X_s[0], y_s[0], D, height+1)
            DT.children['<='+str(s)] = t0
            t1 = generate_tree(X_s[1], y_s[1], D, height+1)
            DT.children['>'+str(s)] = t1
            return DT

def decision_result(x,y,T):
    """返回验证样本(x,y)在决策树中的判定结果是否正确"""
    if T.d == -1:
        return T.kind == y
    else:
        if x[T.d] <= T.s:
            return decision_result(x,y,T.children['<='+str(T.s)])
        else:
            return decision_result(x,y,T.children['>'+str(T.s)])

def decision_precision(X,y,T):
    res = 0
    for x,y_ in zip(X,y):
        res += int(decision_result(x,y_,T))
    return res / len(y)

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
        pass
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

def print_tree(T, layer=0):
    """合理输出决策树"""
    if T.d == -1:
        print(' '*10*layer, 'leaf (result :{})'.format(res_names[T.kind]), sep='')
    else:
        print(' '*10*layer, 'branch  划分标准: {}'.format(X_names[T.d]),sep='')
        for d_value, tree in T.children.items():
            print(' '*10*(layer+1), '{}'.format(d_value), sep='')
            print_tree(tree, layer+1)


if __name__ == '__main__':
    D = set(range(X.shape[1]))
    T = generate_tree(X,y,D,0)
    print('*'*80)
    print('\n未剪枝：Accuracy: {:.3f}\n'.format(decision_precision(X_v,y_v,T)))
    print_tree(T)
    print('\n','*'*80,sep='')
    post_pruning(X_v,y_v,T)
    print('\n后剪枝：Accuracy: {:.3f}\n'.format(decision_precision(X_v,y_v,T)))
    print_tree(T)
    print('\n','*'*80,sep='')



