from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import math
import torch 
import os
from torch.autograd import Variable
import time
import torch.utils.data as Data
from torch import autograd
from torch_geometric.data import InMemoryDataset, Data
from torch import Tensor
from torch.nn import Parameter as Param
from torch_geometric.nn.conv import MessagePassing 
import inspect
from torch.nn import Parameter
from torch_geometric.utils import scatter_
import torch.nn.functional as F
from torch_geometric.data import DataLoader

torch.autograd.set_detect_anomaly(True)

torch.set_printoptions(precision=None, threshold=5000, edgeitems=None, linewidth=None, profile=None)


class Gen_Data(torch.nn.Module):
    def __init__(self, SNR, num, batch_num):
        super(Gen_Data, self).__init__()
        SNR = torch.Tensor(SNR).unsqueeze(0)
        self.Sigma = ((0.36 / (10 ** (SNR / 10)) ** 0.5).repeat(1, num)).repeat(1, batch_num)
        
    def modulate(self, inverse, x):
        if not inverse:
            return 0.6 - 1.2 * x
        else:
            return (0.6 - x) / 1.2
    
    def AWGN(self, x):
        x_prime = self.modulate(0, x)
        noise = torch.normal(0.0, self.Sigma.t().mm(torch.ones((1, x.size()[1])))) #minibatch x n
        y_prime = x_prime + noise
        post = self.get_post(y_prime)
        y = self.modulate(1, y_prime)
        return y, post
    
    def get_post(self, x):
        post = 2 * x.mul(1 / ((self.Sigma ** 2).t().repeat(1, x.size()[1])))
        return post
    

class CustomDataset(InMemoryDataset):
    def __init__(self, H, dataset, x, fac_num, transform = None):
        super(CustomDataset, self).__init__('.', transform, None, None)
        adj = H.to_sparse()
        edge_index = adj._indices()
        data_list = []
        
        for row in dataset:
            data = Data(edge_index=edge_index)
            row = torch.cat([row.unsqueeze(0).t(), torch.zeros(fac_num, 1)], dim=0)
            data.x = row
            data.y = x.t()
            data_list.append(data)
            
        self.data, self.slices = self.collate(data_list)
        
    def _download(self):
        return

    def _process(self):
        return

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__) 


torch.autograd.set_detect_anomaly(True)
num = 20
batch_num = 30
train_num = 1
SNR1 = [1,2,3,4,5,6]
SNR2 = [6] * 6
H_BCH = torch.from_numpy(np.loadtxt('BCH(63,45).txt')).float().t()
H_LDPC = torch.Tensor([[1,1,1,0,0,0,1,0,0,0,0],
                       [0,0,0,1,1,1,0,1,0,0,0],
                       [1,0,0,1,0,0,0,0,1,0,0],
                       [0,1,0,0,1,0,0,0,0,1,0],
                       [0,0,1,0,0,1,0,0,0,0,1]]).t()
H = H_BCH
#H = H_LDPC
data1 = Gen_Data(SNR1, num, batch_num)
data2 = Gen_Data(SNR2, num, batch_num)
x = torch.ones((1, 63))
#x = torch.zeros((1, 11))
y1, post1 = data1.AWGN(x)
y2, post2 = data2.AWGN(x)
train_dataset = CustomDataset(H, post1, x, H.size(1))
test_dataset = CustomDataset(H, post2, x, H.size(1))
BATCH_SIZE = 120
lr = 1e-3
Nc = 5
rows, cols = H.size(0), H.size(1)
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=False)


class GatedGraphConv(MessagePassing):
    def __init__(self, flow, aggr='add', bias=True):
        super(GatedGraphConv, self).__init__(aggr, flow)

        self.mlp = torch.nn.Sequential(torch.nn.Linear(2, 10),
                       torch.nn.ReLU(),
                       torch.nn.Linear(10, 1))
        self.rnn = torch.nn.GRUCell(1, 1, bias=bias)

    def forward(self, x, edge_index):
        '''
        GGC behaviour need to be modified to fellow TRP decoding, which will have 2 phases of iteration; also note that phase2 use 
        the proir knoledge to update rather than the last hidden state of x
        '''
        h = x if x.dim() == 2 else x.unsqueeze(-1)
            
        index_one, index_two = [], []
        
        for i in range(0, len(h), rows + cols):
            for j in range(rows): index_one.append(i+j)
            for j in range(cols): index_two.append(rows+i+j) #last #facotrs nodes indexin a mini-batch
                
        if self.flow == 'source_to_target': idx = index_two
        else: idx = index_one

        m = self.propagate(edge_index=edge_index, size=((rows+cols) * BATCH_SIZE, (rows+cols) * BATCH_SIZE), x=x)
        h[idx] = self.rnn(h[idx].clone(), m[idx].clone())
        
        return h
    
    def message(self, x_i, x_j):
        tmp = torch.cat([x_j, x_i], dim=1)
        
        return self.mlp(tmp)
        
def a_p(grad):
    print(grad.sum().item())
    
    
class GNNI(torch.nn.Module):
    def __init__(self, Nc):
        super(GNNI, self).__init__()
        
        self.Nc = Nc
        self.ggc1 = GatedGraphConv("source_to_target")
        self.ggc2 = GatedGraphConv("target_to_source")
    
    def forward(self, data):
        '''
        this part need to sum up the message and return
        '''
        x = data.x
        edge_index = torch.cat([data.edge_index[0].unsqueeze(0), data.edge_index[1].unsqueeze(0).add(rows)], dim=0)
        
        for i in range(self.Nc):            
            x = self.ggc1(x, edge_index)
            x = self.ggc2(x, edge_index)

        x = torch.sigmoid(-1 * x)
        x = torch.clamp(x, 1e-7, 1-1e-7)
        
        return x


class LossFunc(torch.nn.Module):
    def __init__(self, H):
        super(LossFunc, self).__init__()
        
        self.a, self.b = max(H.size()), min(H.size())

    def forward(self, pred, y):
        res = pred[0 : self.a].clone()
        
        for i in range(self.a + self.b, len(pred), self.a+self.b):
            res = torch.cat([res, pred[i : i+self.a].clone()], dim=0)
        loss_b = torch.log(res)
        loss_a = torch.log(1 - res)
        loss = (1 - y).mul(loss_a) + y.mul(loss_b)
        loss = torch.sum(loss) / (-1 * torch.numel(loss))
#        print(loss)
        
        return loss


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
decoder = GNNI(Nc).to(device)
optimizer = torch.optim.Adam(decoder.parameters(), lr, weight_decay=5e-4)
criterion = LossFunc(H)


def train(epoch):
    decoder.train()
    f = open('./training_loss_for_classic.txt','a')
    
    for datas in train_loader:
        datas = datas.to(device)
        optimizer.zero_grad()
        loss = criterion(decoder(datas), datas.y)
        loss.backward()
#        for p in decoder.parameters():
#            print(p.grad.sum().item())
        optimizer.step()
        
    if epoch % 6 == 0:
        f.write(' %.15f ' % (loss.item()))
        torch.save(decoder.state_dict(), './model/decoder_parameters_epoch%d.pkl' % (epoch))
        
    f.close()
    
    return


def test(decoder):
    decoder.eval()
    loss = 0
    i = 0
    
    for datas in test_loader:
        datas = datas.to(device)
        pred = decoder(datas)
        loss += criterion(pred, datas.y).item()
        i += 1
        
    return loss/i


if __name__ == '__main__':
    training = 0
    load = 1
    if training:
        for epoch in range(1, 151):
            train(epoch)
            test_acc = test(decoder)
            print('Epoch: {:03d}, Test Acc: {:.15f}'.format(epoch, test_acc))
#    
#    if load:
#        for i in range(6, 151, 6):
#            print(i)
#            f = open('./test_loss_for_classic.txt','a')
#            decoder_a = GNNI(Nc).to(device)
#            decoder_a.load_state_dict(torch.load('./model/decoder_parameters_epoch%d.pkl' % (i)))
#            loss = test(decoder_a)
#            print(loss)
#            f.write(' %.30f ' % (loss))
#            
#            f.close()

    if load:
        f = open('./test_loss_for_trained_model.txt','a')
        decoder_b = GNNI(Nc).to(device)
        decoder_b.load_state_dict(torch.load('./decoder_parameters.pkl'))
        loss = test(decoder_b)
        print(loss)
        f.write(' %.30f ' % (loss))
        
        f.close()

        