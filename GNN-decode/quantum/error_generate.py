import math
import random
import numpy as np
import matplotlib.pyplot as plt
import torch 
import os
import sys
import time

np.set_printoptions(threshold = 1e6)

'''first part: toric code stablilizer'''
'''需要首先得到stabilizer， L = 4、6、8....18，找到lattice的表达形式，推导stablilizer，及logical operators'''
'''H和stabilizer都为一个（2*L^2-2）*（4*L^2）的矩阵，令其为上Z，下X;左Z，右X的矩阵'''

def encoder(H, L, c, v, a):
    for i in range(int(L/2)):
        ci,vi = c, v
        for j in range(int(L/2)):
            #print(c,v)
            if ((v<2*L**2) and (c==(L**2-1))) or (c==(2*L**2-2)):pass
            else:H[c, v] = a
            if (c<(L**2-1)and(c%L+2<L))or((c>L**2-2)and(((c-L**2+1)%L+2)<L)and(v>2*L**2-1)):c=c+2
            else:c = c+2-L
            if ((v%L+2)<L):v = v+2
            else:v = v+2-L
            
        if v<2*L**2:
            c = (ci+2*L)%(L**2)
            v = (vi+4*L)%(2*L**2)
        else:
            c = (ci-(L**2-1)+2*L)%(L**2)+(L**2-1)
            v = (vi+4*L-2*L**2)%(2*L**2)+2*L**2

    return H


#generator 去掉了最后一个
def generate_PCM(k, L):
    j = 0
    H = np.zeros([(2 * L * L - 2), (4 * L * L)])
    H_prime = np.zeros([(2 * L * L - 2), (4 * L * L)])
    orders = [] #follow the order of up, left, right, down for Z and X
    H_one = np.zeros([(2 * L * L - 2), (4 * L * L)])
    elements = []
    off_c = L**2-1
    off_v = 2*L**2

    for i in range(8):
        elements.append((L**2-L, 0, 0.2))
        elements.append((L**2-L+1, 1, 1))
        elements.append((L-1, L, 2))
        elements.append((1, L+1, 3))
        elements.append((0, L+1, 4))
        elements.append((2, L+2, 5))
        elements.append((0, L*2, 6))
        elements.append((L, L*2, 7))
        elements.append((1, L*2+1, 8))
        elements.append((1+L, L*2+1, 9))
        elements.append((2*L-1, L*3, 10))
        elements.append((L+1, L*3+1, 11))
        elements.append((L, L*3+1, 12))
        elements.append((L+2, L*3+2, 13))
        elements.append((L*2, L*4, 14))
        elements.append((L*2+1, L*4+1, 15))

        elements.append((off_c+L**2-L, off_v+2*L**2-L, 16.2))
        elements.append((off_c+L**2-L+1, off_v+2*L**2-L+1, 17))
        elements.append((off_c+L-1, off_v+L-1, 18))
        elements.append((off_c, off_v, 19))
        elements.append((off_c+1, off_v, 20))
        elements.append((off_c+2, off_v+1, 21))
        elements.append((off_c, off_v+L, 22))
        elements.append((off_c+L, off_v+L, 23))
        elements.append((off_c+1, off_v+L+1, 24))
        elements.append((off_c+L+1, off_v+L+1, 25))
        elements.append((off_c+2*L-1, off_v+3*L-1, 26))
        elements.append((off_c+L, off_v+2*L, 27))
        elements.append((off_c+L+1, off_v+2*L, 28))
        elements.append((off_c+L+2, off_v+2*L+1, 29))
        elements.append((off_c+2*L, off_v+3*L, 30))
        elements.append((off_c+2*L+1, off_v+3*L+1, 31))

    for e in elements:
        H_one = encoder(H_one, L, e[0], e[1], e[2])

    for i in range(int(k / 2)):
        H[i][j] = H[i][j + L] = H[i][(j + 2 * L) % (2 * L * L)] = 1
        H[i + int(k / 2)][2 * L * L + j] = H[i + int(k / 2)][2 * L * L + j + L] \
        = H[i + int(k / 2)][2 * L * L + (j - L) % (2 * L * L)]  = 1
        
        #append up and left for Z
#        orders.append([i, j])
#        orders.append([i, j+L])
        H_prime[i][j] = 0.2
        H_prime[i][j+L] = 1
        #append right and down for Z
        if (j + L + 1) % L == 0:
            H[i][j + 1] = 1
#            orders.append([i, j+1])
            H_prime[i][j+1] = 2
        else:
            H[i][j + L + 1] = 1
#            orders.append([i, j+L+1])
            H_prime[i][j+L+1] = 2
#        orders.append([i, (j + 2 * L) % (2 * L * L)])
        H_prime[i][(j + 2 * L) % (2 * L * L)] = 3
        
        #append up and left for X
#        orders.append([i + int(k / 2), 2 * L * L + (j - L) % (2 * L * L)])
        H_prime[i + int(k / 2)][2 * L * L + (j - L) % (2 * L * L)] = 4.2
        if j % L != 0:
            H[i + int(k / 2)][2 * L * L + j - 1] = 1
            H_prime[i + int(k / 2)][2 * L * L + j - 1] = 5
#            orders.append([i + int(k / 2), 2 * L * L + j - 1]) #left
        else:
            H[i + int(k / 2)][2 * L * L + (j - 1 + L)] = 1
#            orders.append([i + int(k / 2), 2 * L * L + (j - 1 + L)])
            H_prime[i + int(k / 2)][2 * L * L + (j - 1 + L)] = 5
            
        #append right and down for X
        H_prime[i + int(k / 2)][2 * L * L + j] = 6
        H_prime[i + int(k / 2)][2 * L * L + j + L] = 7
        orders.append([i + int(k / 2), 2 * L * L + j]) #right
        orders.append([i + int(k / 2), 2 * L * L + j + L]) #down
        
        
        if (j + 1) % L == 0:
            j = j + L
        j += 1
        
    return H, H_one

'''得到Log'''
'''
假设H是parity check matrix。

用高斯消元法找到H的nullspace，也就是所有和H内积（别忘了用辛度规和mod2）为零的向量组成的子空间。Nullsapce里有所有的stabilizer generator
和所有的logical X和logical Z，维度是M+2K。
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
        H_prime = self.H.clone()
        exchange = []
        
        for i in range(self.rows):
            if H_prime[i, i] != 1:
                for j in range(i, self.cols):
                    if H_prime[i, j] == 1:
                        H_prime[:, [i, j]] = H_prime[:, [j, i]]
                        exchange.append((i,j))
            
            for j in range(self.rows):
                if (H_prime[j ,i] == 1) and (i != j):
                    
                    H_prime[j, :] = (H_prime[i, :] + H_prime[j, :]) % 2
#            if H_prime[i, i] != 1:
#                print("f")
                
        exchange.reverse()
        
#        for i in range(self.rows):
#            for j in range(self.rows):
#                if (j != i) and (H_prime[i][j] == 1):
#                    print(i, j)
#        for i in range(self.rows):
#            if H_prime[i, i] == 0:
#                print(i,'failed')
#        print(H[:, 0:self.rows])
#        print(H_prime)
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
    
    def symplectic_product(self, a, b):
        a = a if a.dim() == 2 else a.unsqueeze(0)
        b = b if b.dim() == 2 else b.unsqueeze(0)
        rows, cols = b.shape
#        print(a.shape, b.shape)
        return np.dot(a, np.concatenate([b[:, int(cols / 2) : cols].clone(), b[:, 0 : int(cols / 2)].clone()], axis = 1).T) % 2
        
    
    '''
    那么怎么区分logical和stabilizer呢？
    遍历nullspace的basis，每当找到一对辛内积为1的basis vectors，就得到了一对logical X 和Z。这时候可以把这一对存下来，剩下需要遍历的basis
    数目就 -2了。
    在继续遍历之前，我们要确保剩下的basis中不含有刚刚找到的X和Z。做法就是，用刚刚找到的X和所有剩下的basis做内积。如果结果是1，说明这个basis
    含有相应的Z，只要把Z加上就可以去掉（一个向量加自己 = 0 mod2）。用同样的方法去掉X。
    如此循环直到找到K对logical。
    '''
    def get_logical(self, H_prep):
        self.H_prep_p = H_prep.clone()
        stab = H_prep.clone()
        rows, cols = self.H_prep_p.shape
        logical = []
        
        for i in range(rows):
            if not any([(self.H_prep_p[i, :] == logical_).all() for logical_ in logical]):
                
                for j in range(i + 1, rows):
                    if not any([(self.H_prep_p[j, :] == logical_).all() for logical_ in logical]):
                        if self.symplectic_product(self.H_prep_p[i, :], self.H_prep_p[j, :]) == 1:
                            logical.append(self.H_prep_p[i, :])
                            logical.append(self.H_prep_p[j, :])
                            
                            for k in range(j + 1, rows):
                                if not any([(self.H_prep_p[k, :] == logical_).all() for logical_ in logical]):
                                    if self.symplectic_product(self.H_prep_p[i, :], self.H_prep_p[k, :]) == 1:
                                        self.H_prep_p[k, :] = (self.H_prep_p[k, :] + self.H_prep_p[j, :]) % 2
                                        
                            for m in range(i + 1, rows):
                                if not any([(self.H_prep_p[m, :] == logical_).all() for logical_ in logical]):
                                    if self.symplectic_product(self.H_prep_p[j, :], self.H_prep_p[m, :]) == 1:
                                        self.H_prep_p[m, :] = (self.H_prep_p[m, :] + self.H_prep_p[i, :]) % 2
                                 
                            break
        
        for i in range(rows):
            for item in logical:
                if all([(stab[i, :] == item).all()]):
                    stab[i, :] -= item
                
        log = logical[0].unsqueeze(0)
        
        for i in range(1, len(logical)):
            log = torch.cat([log, logical[i].unsqueeze(0)], dim=0)
            
        return log, stab
        
    
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
#        err = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0, 0., 1., 0., 0., 0., 0., 0., 0., 0.,
#            1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#         0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
#        prior[0][5] = 10
        syn_prime = (np.dot(H.t(), err.T) % 2).T
        syn = syn_prime
        for i in range(len(syn)):
            syn[i] = (-1) ** syn_prime[i]
#        print(prior)
        dataset.append(torch.from_numpy(np.concatenate([prior, syn], axis = 1)))
        dataset.append(torch.from_numpy(err))
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
    
    L = 4
    H = generate_PCM(2 * L * L - 2, L)
#    m = H.copy()
    P = [0.01]
#    dataset = gen_syn(P, L, H, 120000)
#    print(sys.getsizeof(dataset))
    gen_syn(P, L, H, 10)
    
    h = H_Prep(H)
    
    H_p = h.get_H_Prep()
#    print(H)
    print(h.symplectic_product(H_p, H).sum())
#    print(H[0 : 8, 0 : 18])
#    print(H[8 : 16, 18 : 36])

#
#    print(H_stan[8 : 16, 26 : 34])
#48:96, 146:194    
#35:70,107:142
##加上break 6错，不加7错
        
