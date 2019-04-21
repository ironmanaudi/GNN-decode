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
import error_generate


torch.autograd.set_detect_anomaly(True)

torch.set_printoptions(precision=None, threshold=5000, edgeitems=None, linewidth=None, profile=None)


class CustomDataset(InMemoryDataset):
    def __init__(self, H, dataset, transform = None):
        super(CustomDataset, self).__init__('.', transform, None, None)
        adj = H.to_sparse()
        edge_index = adj._indices()
        data_list = []
        
        for i in range(0, len(dataset), 2):
            data = Data(edge_index=edge_index)
            data.x = dataset[i].t()
            data.y = dataset[i+1].t()
            data_list.append(data)
            
        self.data, self.slices = self.collate(data_list)
        
    def _download(self):
        return

    def _process(self):
        return

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__) 


torch.autograd.set_detect_anomaly(True)
L = 4
P1 = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
P2 = [0.01]
H = torch.from_numpy(error_generate.generate_PCM(2 * L * L - 2, L)).float() #64,30
h_prep = error_generate.H_Prep(H)
H_prep = torch.from_numpy(h_prep.get_H_Prep()).float()
BATCH_SIZE = 120
lr = 1e-3
Nc = 5
run1 = 2400
run2 = 1200
dataset1 = error_generate.gen_syn(P1, L, H.t(), run1)
dataset2 = error_generate.gen_syn(P2, L, H.t(), run2)
train_dataset = CustomDataset(H, dataset1)
test_dataset = CustomDataset(H, dataset2)
rows, cols = H.size(0), H.size(1)
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=False)

        
def a_p(grad):
    print(abs(grad).max().item(), grad.sum().item(), abs(grad).sum().item())
    

class GatedGraphConv(MessagePassing):
    def __init__(self, flow, aggr='add', bias=True):
        super(GatedGraphConv, self).__init__(aggr, flow)
        
        self.mlp = torch.nn.Sequential(torch.nn.Linear(2, 20),
                       torch.nn.ReLU(),
                       torch.nn.Linear(20, 1))
        self.rnn = torch.nn.GRUCell(1, 1, bias=bias)


    def forward(self, x, edge_index):
        '''
        GGC behaviour need to be modified to fellow BP decoding, which will have 2 phases of iteration; also note that phase2 use 
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
#        print('a', abs(x).max().item(), abs(x).min().item())
#        x.register_hook(a_p)
        x = torch.sigmoid(x)
        
#        x = torch.clamp(x, 1e-10, 1-1e-10)
        
        return x


class LossFunc(torch.nn.Module):
    def __init__(self, H, H_prep):
        super(LossFunc, self).__init__()
        
        self.a, self.b = max(H.size()), min(H.size())
#        self.H_prep = Variable(H_prep).cuda()
        self.H_prep = H_prep
        
    def forward(self, pred, y):
        res = pred[0 : self.a].clone()
        tmp = y[0 : self.a].clone()
        
        for i in range(self.a + self.b, len(pred), self.a+self.b):
            res = torch.cat([res, pred[i : i+self.a].clone()], dim=1)
            
        for i in range(self.a, len(y), self.a):
            tmp = torch.cat([tmp, y[i : i+self.a]], dim=1)
            
#        res.register_hook(a_p)
        
        res = torch.matmul(self.H_prep, res + tmp)
        
        res = abs(torch.sin(res * math.pi / 2)).sum()
        
        print(res / (BATCH_SIZE * (2 * L ** 2 - 2)))
        
        return res
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
decoder = GNNI(Nc).to(device)
optimizer = torch.optim.Adam(decoder.parameters(), lr, weight_decay=5e-4)
criterion = LossFunc(H, H_prep)


def train(last):
    decoder.train()
    
    for datas in train_loader:
        datas = datas.to(device)
        optimizer.zero_grad()
        criterion(decoder(datas), datas.y).backward()
        
#        for p in decoder.parameters():
#            print(p.grad.sum().item())
        
        optimizer.step()
        
    if last == 1:
        torch.save(decoder.state_dict(), './model/decoder_parameters.pkl')
    return


def test():
    decoder.eval()
    loss = 0
    for datas in test_loader:
        datas = datas.to(device)
        pred = decoder(datas)
        loss += criterion(pred, datas.y).item()
        
    return loss / (run2 * (2 * L ** 2 - 2))


if __name__ == '__main__':
    training = 1
    load = 0
    if training:
        for epoch in range(1, 201):
#            print(epoch)
            last = 0
            if epoch == 200:
                last = 1
            train(last)
            test_acc = test()
            print('Epoch: {:03d}, Test Acc: {:.5f}'.format(epoch, test_acc))
    