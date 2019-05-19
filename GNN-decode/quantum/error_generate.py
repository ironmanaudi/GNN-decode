import math
import random
import numpy as np
import matplotlib.pyplot as plt
import torch 
import os
import sys

np.set_printoptions(threshold = 1e6)

'''first part: toric code stablilizer'''
'''需要首先得到stabilizer， L = 4、6、8....18，找到lattice的表达形式，推导stablilizer，及logical operators'''
'''H和stabilizer都为一个（2*L^2-2）*（4*L^2）的矩阵，令其为上Z，下X;左X，右z的矩阵'''

#generator 去掉了最后一个
def generate_PCM(k, L):
    j = 0
    H = np.zeros([(2 * L * L - 2), (4 * L * L)])
    for i in range(int(k / 2)):
        H[i][j] = H[i][j + L] = H[i][(j + 2 * L) % (2 * L * L)] = 1
        H[i + int(k / 2)][2 * L * L + j] = H[i + int(k / 2)][2 * L * L + j + L] \
        = H[i + int(k / 2)][2 * L * L + (j - L) % (2 * L * L)]  = 1
        if (j + L + 1) % L == 0:
            H[i][j + 1] = 1
        else:
            H[i][j + L + 1] = 1
        if j % L != 0:
            H[i + int(k / 2)][2 * L * L + j - 1] = 1
        else:
            H[i + int(k / 2)][2 * L * L + (j - 1 + L)] = 1
        if (j + 1) % L == 0:
            j = j + L
        j += 1
    return H

'''得到Log'''
'''
假设H是parity check matrix。

用高斯消元法找到H的nullspace，也就是所有和H内积（别忘了用辛度规和mod2）为零的向量组成的子空间。Nullsapce里有所有的stabilizer generator
和所有的logical X和logical Z，维度是M+2K。
那么怎么区分logical和stabilizer呢？
遍历nullspace的basis，每当找到一对辛内积为1的basis vectors，就得到了一对logical X 和Z。这时候可以把这一对存下来，剩下需要遍历的basis
数目就 -2了。
在继续遍历之前，我们要确保剩下的basis中不含有刚刚找到的X和Z。做法就是，用刚刚找到的X和所有剩下的basis做内积。如果结果是1，说明这个basis
含有相应的Z，只要把Z加上就可以去掉（一个向量加自己 = 0 mod2）。用同样的方法去掉X。
如此循环直到找到K对logical。
'''
'''
null space over GF(2) reference:
https://math.stackexchange.com/questions/130207/finding-null-space-basis-over-a-finite-field
'''
class H_Prep():
    def __init__(self, H):
        self.H = H
        self.rows = int(H.shape[0])
        self.cols = int(H.shape[1])
#        print(self.rows, self.cols)
        self.H_prep = H
        
    def Get_Identity(self):
        H_prime = self.H
        exchange = []
        for i in range(self.rows):
            if H_prime[i, i] != 1:
                for j in range(i, self.cols):
                    if H_prime[i, j] == 1:
                        H_prime[:, [i, j]] = H_prime[:, [j, i]]
                        exchange.append((i,j))
        exchange.reverse()
#        for i in range(self.rows):
#            if H_prime[i, i] == 0:
#                print(i,'failed')
#        print(H[:, 0:self.rows])
        return H_prime, exchange
    
    def get_H_Prep(self):
        self.H_prep, exchange = self.Get_Identity()
        self.H_prep = np.concatenate([self.H_prep[:, self.rows : self.cols], np.eye(self.cols - self.rows)], axis = 0)
        for item in exchange:
            self.H_prep[[item[0], item[1]], :] = self.H_prep[[item[1], item[0]], :]
        self.H_prep = np.concatenate([self.H_prep[int(self.cols / 2) : self.cols, :], \
                                                  self.H_prep[0 : int(self.cols / 2), :]], axis = 0)
        self.H_prep = self.H_prep.T
        return self.H_prep
    
    def symplectic_product(a, b):
        rows, cols = a.shape
        return np.dot(a, np.concatenate([b[:, int(cols / 2) : cols], b[:, 0 : int(cols / 2)]], axis = 1).T) % 2
    
    def get_logical(self):
        self.H_prep = self.get_H_Prep()
        rows, cols = self.H_prep.shape
        logical = []
        for i in range(rows):
            if self.H_prep[i, :] not in logical:
                for j in range(i + 1, rows):
                    if self.H_prep[j, :] not in logical:
                        if self.symplectic_product(self.H_prep[i, :], self.H_prep[j, :]) == 1:
                            logical.append(self.H_prep[i, :])
                            logical.append(self.H_prep[j, :])
                            for k in range(j + 1, rows):
                                if self.H_prep[k, :] not in logical:
                                    if self.symplectic_product(self.H_prep[i, :], self.H_prep[k, :]) == 1:
                                        self.H_prep[k, :] += self.H_prep[j, :]
                            for m in range(i + 1, rows):
                                if self.H_prep[m, :] not in logical:
                                    if self.symplectic_product(self.H_prep[j, :], self.H_prep[m, :]) == 1:
                                        self.H_prep[m, :] += self.H_prep[i, :]
                            break
        logical = np.array(logical)
        return logical
        
'''数据集产生:error generate，要有原始的error记录及相应的syndrome'''
def gen_syn(P, L, H, run):
    dataset = []
    err = np.zeros((1, 4 * L * L))
    rows, cols = H.shape
    for i in range(run):
        p = random.sample(P, 1)[0]
        prior = np.full((1, 4 * L * L), math.log((1 - p) / p))
        for j in range(2 * L * L):
            a, b = np.random.random(), np.random.random()
            if a < p:
                err[0, j] = 1 #X error
            if b < p:
                err[0, j + 2 * L * L] = 1 #Z error
        syn_prime = (np.dot(H.t(), err.T) % 2).T
        syn = syn_prime
        for i in range(len(syn)):
            syn[i] = (-1) ** syn_prime[i]
        dataset.append(torch.from_numpy(np.concatenate([prior, syn], axis = 1)).float())
        dataset.append(torch.from_numpy(err).float())
        err = np.zeros((1, 4 * L * L))
    return dataset

#def data_base(L, p, train, run):
#    H = generate_PCM(2 * L * L - 2, L)
#    if train:
#        s = 'train'
#    else:
#        s = 'test'
#    for i in range(run):
#        Syn, Err = gen_syn(p, L, H)
#        path = '.\\data_base\\L = %d_p = %f_%s' % (L, p, s)
#        if not os.path.exists(path):
#            os.makedirs(path)
#        filename = path + '\\L = %d_p = %f run = %d_%s.txt' % (L, p, i, s)
#        Syn, Err = gen_syn(p, L, H, run)
#        f = open(filename, 'w')
#        f.write(Err)
#        f.close()

if __name__ == '__main__':
    
    L = 12
    H = generate_PCM(2 * L * L - 2, L)
    P = [0.01]
    dataset = gen_syn(P, L, H, 120000)
    print(sys.getsizeof(dataset))
#    gen_syn(P, L, H, 10)
#    h = H_Prep(H)
#    H_prep = h.get_H_prep()
#    print(H_prep)
#    print(H[0 : 8, 0 : 18])
#    print(H[8 : 16, 18 : 36])

#
#    print(H_stan[8 : 16, 26 : 34])
#48:96, 146:194    
#35:70,107:142
##加上break 6错，不加7错
        
