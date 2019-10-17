import numpy as np
from collections import Counter

X = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([4.50,4.75,4.91,5.34,5.80,7.05,7.90,8.23,8.70,9.00])

class RegressionTree:
    """定义回归树类，属性有预估值（value）| 划分值（s）| 左右子树（children）"""
    def __init__(self, value, s=float('-inf')):
        self.value = value
        self.s = s
        self.children = []

def square_error(y):
    """求解平方误差值"""
    return sum((y-y.mean())**2)

def split(X,y,s):
    """根据s的值对X和y进行划分"""
    index_a = (X <= s)
    index_b = (X > s)
    return X[index_a], X[index_b], y[index_a], y[index_b]

def try_split(X,y):
    """找出针对X和y的最佳划分"""
    best_se = float('inf')
    best_s = float('-inf')
    #本题中提供的X值为有序排列，为了应对多种情况，假设其无序
    sorted_index = np.argsort(X)
    for i in range(1, len(X)):
        if X[sorted_index[i]] != X[sorted_index[i-1]]:
            s = (X[sorted_index[i]] + X[sorted_index[i-1]]) / 2
            X_l, X_r, y_l, y_r = split(X, y, s)
            se = square_error(y_l) + square_error(y_r)
            if se < best_se:
                best_se, best_s = se, s
    return best_se, best_s

def generate_RT(X,y,depth):
    c = y.mean()
    if depth <= 1:
        return RegressionTree(c)
    elif len(Counter(X)) <= 1:
        return RegressionTree(c)
    else:
        se, s = try_split(X, y)
        X_l, X_r, y_l, y_r = split(X, y, s)
        c_l, c_r = y_l.mean(), y_r.mean()
        RT = RegressionTree(c, s)
        t1 = generate_RT(X_l, y_l, depth-1)
        t2 = generate_RT(X_r, y_r, depth-1)
        RT.children = [t1, t2]
        return RT
    
def print_RT(RT, layer=0):
    if RT.s == float('-inf'):
        print(' '*5*layer, 'leaf(c = {})'.format(RT.value),sep='')
    else:
        print(' '*5*layer, 'branch', sep='')
        print(' '*5*(layer+1), 'X <= {}'.format(RT.s), sep='')
        print_RT(RT.children[0], layer+1)
        print(' '*5*(layer+1), 'X > {}'.format(RT.s), sep='')
        print_RT(RT.children[1], layer+1)

if __name__ == '__main__':
	RT = generate_RT(X,y,3)
	print_RT(RT)