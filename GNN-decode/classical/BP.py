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
        
        if self.flow == 'target_to_source':
            out = torch.clamp(out, -10, 10)
            out = torch.tanh(out / 2)
            Coeff = torch.where(out < 0, torch.ones(out.size()).cuda(), torch.zeros(out.size()).cuda())
            out = abs(out)
            out = torch.clamp(out, 1e-7, 1e10)
            out = torch.log(out)
#            print('a',out.t())
            out = scatter_(self.aggr, out, edge_index[j], dim_size=size[i])[edge_index[j]] - out
#            print('b',out.t())
            Coeff = scatter_(self.aggr, Coeff, edge_index[j], dim_size=size[i])[edge_index[j]] - Coeff
            Coeff = torch.cos(math.pi * Coeff)
            out = torch.exp(out).mul(Coeff)
            out = torch.clamp(out, -1+1e-7, 1-1e-7)
#            out = torch.log(1 + out) - torch.log(1 - out)
            out = torch.log((1 + out) / (1 - out))
        else:
            out = scatter_(self.aggr, out, edge_index[j], dim_size=size[i])[edge_index[j]] - out
            out = out + extra[edge_index[j]]
#            print(extra[edge_index[j]].t())
        out = self.update(out, *update_args)

        return out


    def message(self, x_j):  # pragma: no cover
        
        return x_j


    def update(self, aggr_out):  # pragma: no cover

        return aggr_out


class Gen_Data(torch.nn.Module):
    def __init__(self, SNR, num):
        super(Gen_Data, self).__init__()
        SNR = torch.Tensor(SNR).unsqueeze(0)
        self.Sigma = ((1 / (10 ** (SNR / 10)) ** 0.5).repeat(1, num))
#        self.Sigma = ((0.36 / (10 ** (SNR / 10)) ** 0.5).repeat(1, num))
        
    def modulate(self, inverse, x):
        if not inverse:
            return 1 - 2 * x
        else:
            return (1 - x) / 2
    
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


num = 100
SNR2 = [6]*6
H_BCH = torch.from_numpy(np.loadtxt('BCH(63,45).txt')).float().t()
#H_LDPC = torch.Tensor([[1,1,1,0,0,0,1,0,0,0,0],
#                       [0,0,0,1,1,1,0,1,0,0,0],
#                       [1,0,0,1,0,0,0,0,1,0,0],
#                       [0,1,0,0,1,0,0,0,0,1,0],
#                       [0,0,1,0,0,1,0,0,0,0,1]]).t()
H_LDPC = torch.Tensor([[0,1,0,1,1,0,0,1],
                       [1,1,1,0,0,1,0,0],
                       [0,0,1,0,0,1,1,1],
                       [1,0,0,1,1,0,1,0]]).t()
H = H_BCH
#H = H_LDPC
data2 = Gen_Data(SNR2, num)
x = torch.zeros((1, 63))
#x = torch.zeros((1, 8))
#x = torch.zeros((1, 11))
#x = torch.Tensor([[1,0,0,1,0,1,0,1]])
y2, post2 = data2.AWGN(x)
test_dataset = CustomDataset(H, post2, x, H.size(1))
BATCH_SIZE = 120
lr = 3e-4
Nc = 25
rows, cols = H.size(0), H.size(1)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=False)


class GatedGraphConv(MessagePassing):
    def __init__(self, flow, aggr='add', bias=True):
        super(GatedGraphConv, self).__init__(aggr, flow)

        self.flow = flow

    def forward(self, m, edge_index, x=None):
        if x is not None:
            x = x if x.dim() == 2 else x.unsqueeze(-1)
            
        mes = self.propagate(edge_index=edge_index, size=((rows+cols) * BATCH_SIZE, (rows+cols) * BATCH_SIZE), x=m, extra=x)

        return mes
    
    
class GNNI(torch.nn.Module):
    def __init__(self, Nc):
        super(GNNI, self).__init__()
        
        self.Nc = Nc
        self.ggc1 = GatedGraphConv("source_to_target")
        self.ggc2 = GatedGraphConv("target_to_source")
    
    def forward(self, data):
        x = data.x
        edge_index = torch.cat([data.edge_index[0].unsqueeze(0), data.edge_index[1].unsqueeze(0).add(rows)], dim=0)
        m = Variable(torch.zeros((edge_index.size()[1], 1)), requires_grad=False).cuda()
#        print(edge_index)
        for i in range(self.Nc):
            m = self.ggc1(m, edge_index, x)
            m = self.ggc2(m, edge_index)
            
        size=((rows+cols) * BATCH_SIZE, (rows+cols) * BATCH_SIZE)
        res = scatter_('add', m, edge_index[0], dim_size=size[0]) + x
#        res = torch.clamp(res, 1e-7, 1-1e-7)
        
        tmp = res[0 : rows].clone()
        
        for i in range(rows+cols, len(res), rows+cols):
            tmp = torch.cat([tmp, res[i : i+rows].clone()], dim=0)
        
        tmp = torch.sigmoid(-1 * tmp)
        tmp = torch.clamp(tmp, 1e-7, 1-1e-7)
        return tmp


class LossFunc(torch.nn.Module):
    def __init__(self, H):
        super(LossFunc, self).__init__()
        self.H = H.t().cuda()
#        self.a, self.b = max(H.size()), min(H.size())

    def forward(self, pred, y, train):
        loss_a = (1 - y).mul(torch.log(1 - pred)) + y.mul(torch.log(pred))
        loss = torch.sum(loss_a) / (-1 * torch.numel(loss_a))
        
        return loss


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = LossFunc(H)


def test(decoder):
    decoder.eval()
    loss = 0
    i = 0
    
    for datas in test_loader:
#        print(datas.x.t())
        datas = datas.to(device)
        pred = decoder(datas)
        loss += criterion(pred, datas.y, train=0).item()
        i += 1
        
    return loss/i


if __name__ == '__main__':
    decoder_b = GNNI(Nc).to(device)
    loss = test(decoder_b)
    print(loss)