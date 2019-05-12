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
#from torch_geometric.nn.conv import MessagePassing
import inspect
from torch.nn import Parameter
from torch_geometric.utils import scatter_
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import error_generate


torch.autograd.set_detect_anomaly(True)

torch.set_printoptions(precision=None, threshold=5000, edgeitems=None, linewidth=None, profile=None)


special_args = [
    'edge_index', 'edge_index_i', 'edge_index_j', 'size', 'size_i', 'size_j'
]
__size_error_msg__ = ('All tensors which should get mapped to the same source'
                      'or target nodes must be of same size in dimension 0.')


class MessagePassing(torch.nn.Module):
    def __init__(self, aggr='add', flow='source_to_target'):
        super(MessagePassing, self).__init__()

        self.aggr = aggr
        assert self.aggr in ['add', 'mean', 'max']

        self.flow = flow
        assert self.flow in ['source_to_target', 'target_to_source']

        self.__message_args__ = inspect.getfullargspec(self.message)[0][1:]
        self.__special_args__ = [(i, arg)
                                 for i, arg in enumerate(self.__message_args__)
                                 if arg in special_args]
        self.__message_args__ = [
            arg for arg in self.__message_args__ if arg not in special_args
        ]
        self.__update_args__ = inspect.getfullargspec(self.update)[0][2:]

    def propagate(self, edge_index, extra=None, size=None, **kwargs):
        size = [None, None] if size is None else list(size)
        assert len(size) == 2

        i, j = (0, 1) if self.flow == 'target_to_source' else (1, 0)
        ij = {"_i": i, "_j": j}

        message_args = []
        for arg in self.__message_args__:
            if arg[-2:] in ij.keys():
                tmp = kwargs[arg[:-2]]
                if tmp is None:  # pragma: no cover
                    message_args.append(tmp)
                else:
                    idx = ij[arg[-2:]]
                    if isinstance(tmp, tuple) or isinstance(tmp, list):
                        assert len(tmp) == 2
                        if size[1 - idx] is None:
                            size[1 - idx] = tmp[1 - idx].size(0)
                        if size[1 - idx] != tmp[1 - idx].size(0):
                            raise ValueError(__size_error_msg__)
                        tmp = tmp[idx]

                    if size[idx] is None:
                        size[idx] = tmp.size(0)
#                    if size[idx] != tmp.size(0):
#                        raise ValueError(__size_error_msg__)

#                    tmp = torch.index_select(tmp, 0, edge_index[idx])
                    message_args.append(tmp)
            else:
                message_args.append(kwargs[arg])

        size[0] = size[1] if size[0] is None else size[0]
        size[1] = size[0] if size[1] is None else size[1]

        kwargs['edge_index'] = edge_index
        kwargs['size'] = size

        for (idx, arg) in self.__special_args__:
            if arg[-2:] in ij.keys():
                message_args.insert(idx, kwargs[arg[:-2]][ij[arg[-2:]]])
            else:
                message_args.insert(idx, kwargs[arg])

        update_args = [kwargs[arg] for arg in self.__update_args__]

        out = self.message(*message_args)
        
        out = scatter_(self.aggr, out, edge_index[j], dim_size=size[i])[edge_index[j]] - out
        
        if self.flow == 'source_to_target':
            out = out + extra[edge_index[j]]
        else:
#            out = torch.cat([out, extra[edge_index[j]]], dim=1)
            out = out.mul(extra[edge_index[j]])
            
        out = self.update(out, *update_args)

        return out


    def message(self, x_j):  # pragma: no cover
        
        return x_j


    def update(self, aggr_out):  # pragma: no cover

        return aggr_out


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
H = torch.from_numpy(error_generate.generate_PCM(2 * L * L - 2, L)).float().t() #64, 30
h_prep = error_generate.H_Prep(H.t())
H_prep = torch.from_numpy(h_prep.get_H_Prep()).float()
BATCH_SIZE = 120
lr = 1e-3
Nc = 5
run1 = 4800
run2 = 1200
lambda_a = 0.1
dataset1 = error_generate.gen_syn(P1, L, H, run1)
dataset2 = error_generate.gen_syn(P2, L, H, run2)
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
        
        self.flow = flow
        self.mlp1 = torch.nn.Sequential(torch.nn.Linear(1, 20),
                       torch.nn.ReLU(),
                       torch.nn.Linear(20, 1))
        self.mlp2 = torch.nn.Sequential(torch.nn.Linear(1, 20),
                       torch.nn.ReLU(),
                       torch.nn.Linear(20, 1))
        self.rnn = torch.nn.GRUCell(1, 1, bias=bias)

    def forward(self, m, edge_index, x, results):
        '''
        GGC behaviour need to be modified to fellow BP decoding, which will have 2 phases of iteration; also note that phase2 use 
        the proir knoledge to update rather than the last hidden state of x
        '''
        x = x if x.dim() == 2 else x.unsqueeze(-1)
        
        mes = self.propagate(edge_index=edge_index, size=((rows+cols) * BATCH_SIZE, (rows+cols) * BATCH_SIZE), x=m, extra=x)
        m = self.rnn(m, mes)
        
        if self.flow == 'target_to_source':
            results.append(scatter_('add', m, edge_index[0], dim_size=(rows+cols) * BATCH_SIZE))
        
        return m, results
    
    def update(self, x_j):
        if self.flow == 'target_to_source':
            return self.mlp2(x_j)
        else:
            return self.mlp1(x_j)
    
class GNNI(torch.nn.Module):
    def __init__(self, Nc):
        super(GNNI, self).__init__()
        
        self.Nc = Nc
        self.ggc1 = GatedGraphConv("source_to_target")
        self.ggc2 = GatedGraphConv("target_to_source")
        self.mlp = torch.nn.Sequential(torch.nn.Linear(1, 20),
                       torch.nn.ReLU(),
                       torch.nn.Linear(20, 1))
    
    def forward(self, data):
        '''
        this part need to sum up the message and return
        '''
        x = data.x
        edge_index = torch.cat([data.edge_index[0].unsqueeze(0), data.edge_index[1].unsqueeze(0).add(rows)], dim=0)
        m = Variable(torch.zeros((edge_index.size()[1], 1)), requires_grad=False).cuda()
        results = []
        
        for i in range(self.Nc):          
            m, results = self.ggc1(m, edge_index, x, results)
            m, results = self.ggc2(m, edge_index, x, results)        
#        print('a', abs(x).max().item(), abs(x).min().item())
#        x.register_hook(a_p)
            
#        res = scatter_('add', m, edge_index[0], dim_size=(rows+cols) * BATCH_SIZE)
        
        for i in range(len(results)):
            tmp = results[i][0 : rows].clone()
            
            for j in range(rows+cols, len(results[i]), rows+cols):
                tmp = torch.cat([tmp, results[i][j : j+rows].clone()], dim=0)
            
            results[i] = self.mlp(tmp)
            
            results[i] = torch.sigmoid(-1 * results[i])
        
            results[i] = torch.clamp(results[i], 1e-7, 1-1e-7)
        
        return results


class LossFunc(torch.nn.Module):
    def __init__(self, H, H_prep):
        super(LossFunc, self).__init__()
        
        self.a, self.b = max(H.size()), min(H.size())
        self.H_prep = H_prep.cuda()
        
    def forward(self, results, y):
        loss_sum = Variable(torch.zeros((1,1))).cuda()
        tmp = y[0 : self.a].clone()
        
        for i in range(self.a, len(y), self.a):
            tmp = torch.cat([tmp, y[i : i+self.a].clone()], dim=1)
            
        for i in range(len(results)):
            res = results[i][0 : self.a].clone()
            for j in range(self.a, len(y), self.a):
                res = torch.cat([res, results[i][j : j+self.a].clone()], dim=1)
            loss_a = torch.matmul(self.H_prep, res + tmp)
            loss_b = (1 - tmp).mul(torch.log(res)) + res.mul(torch.log(1 - tmp))
            loss_sum += abs(torch.sin(loss_a * math.pi / 2)).sum() + lambda_a * torch.sum(loss_b) / (-1 * torch.numel(loss_b))
#        res.register_hook(a_p)
        
        loss_sum = loss_sum / len(results)
        
        return loss_sum
    

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
        pred = [pred[len(pred)-1]]
        loss += criterion(pred, datas.y).item()
        
    return loss / (run2 * (2 * L ** 2 - 2))


if __name__ == '__main__':
    training = 1
    load = 0
    if training:
        for epoch in range(1, 201):
            last = 0
            if epoch == 200:
                last = 1
            train(last)
            test_acc = test()
            print('Epoch: {:03d}, Test Acc: {:.5f}'.format(epoch, test_acc))