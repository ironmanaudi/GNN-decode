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
        
        if self.flow == 'target_to_source':
            out = torch.tanh(out / 2)
            out = scatter_(self.aggr, out, edge_index[j], dim_size=size[i])[edge_index[j]] - out
        else:
            out = scatter_(self.aggr, out, edge_index[j], dim_size=size[i])[edge_index[j]] - out
        
        if self.flow == 'source_to_target':
#            out = out + extra[edge_index[j]]
            out = torch.cat([out, extra[edge_index[j]]], dim=1)
        else:
            out = torch.cat([out, extra[edge_index[j]]], dim=1)
        
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
L = 8
#lambda_a = 0.5
#P1 = [0.01,0.03,0.05, 0.07, 0.09, 0.11, 0.13]#, 0.15,0.17,0.19,0.20,0.21]
P1 = [0.1]
#P1 = [0.01,0.02,0.03,0.04,0.05,0.06, 0.07,0.08]
#P1 = [0.01,0.04,0.07,0.1,0.13,0.16]
P2 = [0.1]
H = torch.from_numpy(error_generate.generate_PCM(2 * L * L - 2, L)).t() #64, 30
h_prep = error_generate.H_Prep(H.t())
H_prep = torch.from_numpy(h_prep.get_H_Prep())
#print(H_prep.size())
BATCH_SIZE = 128
lr = 3e-4
Nc = 15
run1 = 40960
run2 = 8192
dataset1 = error_generate.gen_syn(P1, L, H, run1)
dataset2 = error_generate.gen_syn(P2, L, H, run2)
train_dataset = CustomDataset(H, dataset1)
test_dataset = CustomDataset(H, dataset2)
rows, cols = H.size(0), H.size(1)
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=False)
#print(h_prep.symplectic_product(H.t(), H_prep).sum())
logical, stab = h_prep.get_logical(H_prep)
logical, stab = logical.cuda(), stab.cuda()


def a_p(grad):
    a = torch.where(abs(grad) < 1e-1, torch.ones(grad.size()).cuda(), torch.zeros(grad.size()).cuda())
#    print(a.sum().item() / grad.numel())
    print(abs(grad).min().item(), abs(grad).max().item(), a.sum().item() / grad.numel())
#    f = open('./grad.txt','w')
#    for i in grad
#        f.write('%f, '%i.item())
#    f.close()
    

class GraphConv(MessagePassing):
    def __init__(self, flow, aggr='add', bias=True):
        super(GraphConv, self).__init__(aggr, flow)
        
        self.flow = flow
        self.mlp = torch.nn.Sequential(torch.nn.Linear(1, 10).double(),
                       torch.nn.ReLU(),
                       torch.nn.Linear(10, 1).double())
        self.mlp_p = torch.nn.Sequential(torch.nn.Linear(1, 10).double(),
                       torch.nn.ReLU(),
                       torch.nn.Linear(10, 1).double())
        self.mlp1 = torch.nn.Sequential(torch.nn.Linear(2, 10).double(),
                       torch.nn.ReLU(),
                       torch.nn.Linear(10, 1).double())
        self.mlp2 = torch.nn.Sequential(torch.nn.Linear(2, 10).double(),
                       torch.nn.ReLU(),
                       torch.nn.Linear(10, 1).double())
        self.rnn1 = torch.nn.GRUCell(1, 1, bias=bias).double()
        self.rnn2 = torch.nn.GRUCell(1, 1, bias=bias).double()
        
    def forward(self, m, edge_index, x):
        '''
        GGC behaviour need to be modified to fellow BP decoding, which will have 2 phases of iteration; also note that phase2 use 
        the proir knoledge to update rather than the last hidden state of x
        '''
        x = x if x.dim() == 2 else x.unsqueeze(-1)
        
        if self.flow == 'target_to_source': m = self.mlp(m)
        else: m = self.mlp_p(m)
        
        mes = self.propagate(edge_index=edge_index, size=((rows+cols) * BATCH_SIZE, (rows+cols) * BATCH_SIZE), x=m, extra=x)
        
        if self.flow == 'target_to_source': mes = self.rnn2(mes, m)
        else: mes = self.rnn1(mes, m)
        
        return mes
    
    def update(self, aggr_out):
        if self.flow == 'target_to_source':
#            aggr_out[:, 0] = self.mlp(aggr_out[:, 0].clone().unsqueeze(1)).squeeze(1)
            
#            return aggr_out[:, 0].clone().unsqueeze(1).mul(aggr_out[:, 1].clone().unsqueeze(1))
            return self.mlp2(aggr_out)
        else:
            return self.mlp1(aggr_out)
#            return self.mlp(aggr_out[:, 0].clone().unsqueeze(1)) + self.mlp1(aggr_out[:, 1].clone().unsqueeze(1))
#            return aggr_out
    
    
class GNNI(torch.nn.Module):
    def __init__(self, Nc):
        super(GNNI, self).__init__()
        
        self.Nc = Nc
        self.ggc1 = GraphConv("source_to_target")
        self.ggc2 = GraphConv("target_to_source")
        self.mlp = torch.nn.Sequential(torch.nn.Linear(1, 10).double(),
                       torch.nn.ReLU(),
                       torch.nn.Linear(10, 1).double())
    
    def forward(self, data):
        '''
        this part need to sum up the message and return
        '''
        x = data.x
        edge_index = torch.cat([data.edge_index[0].clone().unsqueeze(0), data.edge_index[1].clone().unsqueeze(0).add(rows)], dim=0)
        m = Variable(torch.zeros((edge_index.size()[1], 1), dtype = torch.float64), requires_grad=False).cuda()
        
        for i in range(self.Nc):
            m_p = m.clone()
            m = self.ggc1(m, edge_index, x)
#            m.register_hook(a_p)
            m = self.ggc2(m, edge_index, x)
#        print('a', abs(x).max().item(), abs(x).min().item()
        
        size=((rows+cols) * BATCH_SIZE, (rows+cols) * BATCH_SIZE)
        res = self.mlp(scatter_('add', m, edge_index[0], dim_size=size[0]))
        
        tmp = res[0 : rows].clone()
        
        for i in range(rows+cols, len(res), rows+cols):
            tmp = torch.cat([tmp, res[i : i+rows].clone()], dim=0)
        
#        res = self.mlp(tmp)
        
        res = torch.sigmoid(res)
        
        return res


class LossFunc(torch.nn.Module):
    def __init__(self, H, H_prep):
        super(LossFunc, self).__init__()
        
        self.a, self.b = max(H.size()), min(H.size())
#        self.H_prep = Variable(H_prep).cuda()
        self.H_prep = H_prep.cuda()
        
    def forward(self, pred, datas):
        tmp = datas.y[0 : self.a].clone()
        res = pred[0 : self.a].clone()
        syn = datas.x[self.a : self.a+self.b]
        
        for i in range(self.a, len(datas.y), self.a):
#            tmp = torch.cat([tmp, datas.y[i : i+self.a].clone()], dim=1)
            res = torch.cat([res, pred[i : i+self.a].clone()], dim=1)
              
        for i in range(self.a+self.b, len(datas.x), self.a+self.b):
            syn = torch.cat([syn, datas.x[i+self.a : i+self.a+self.b].clone()], dim=1)

#        deg = ((torch.where(res>0.5, torch.ones(res.size()).cuda(), torch.zeros(res.size()).cuda()) + tmp) % 2)
#        deg = torch.where(deg==0, deg, torch.ones(deg.size()).cuda()).sum() / torch.numel(deg)
#        print(deg.item())
#        print('a',(res + tmp).sum().item())
        
#        pred_p = torch.where(res > 0.5, torch.ones(res.size(), dtype=torch.float64).cuda(),\
#                             torch.zeros(res.size(), dtype=torch.float64).cuda()).cuda()
#        err = (pred_p + tmp) % 2
#        print('a', tmp.sum().item()/tmp.numel(), err.sum().item()/tmp.numel())
        loss_p = torch.matmul(H.t().cuda(), tmp + res)
        loss = abs(torch.sin(loss_p * math.pi / 2)).sum()
#        loss_a = torch.matmul(H.t().cuda(), res) + syn
#        loss_b = torch.matmul(logical, tmp + res)
#        loss = abs(torch.sin(loss_a * math.pi / 2)).sum() + abs(torch.sin(loss_b * math.pi / 2)).sum()
#        log_loss = torch.matmul(logical, tmp)
#        log_loss = abs(torch.sin(log_loss * math.pi / 2)).sum()
#        print('b',loss_p.sum().item(), torch.matmul(self.H_prep, err).sum().item())
        
        return loss
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
decoder = GNNI(Nc).to(device)
#decoder.load_state_dict(torch.load('./model1/decoder_parameters_epoch6.pkl'))
optimizer = torch.optim.Adam(decoder.parameters(), lr, weight_decay=5e-4)
#optimizer = torch.optim.RMSprop(decoder.parameters(), lr, alpha=0.9)
criterion = LossFunc(H, H_prep)


def train(epoch):
    decoder.train()
    f = open('./training_loss_for_quantum.txt','a')
    
    for datas in train_loader:
        datas = datas.to(device)
        optimizer.zero_grad()
        loss = criterion(decoder(datas), datas)
        loss.backward()
        
#        for p in decoder.parameters():
#            print(p.grad.sum().item())
        
        optimizer.step()
        
    if epoch % 6 == 0:
        f.write(' %.15f ' % (loss.item()))
        torch.save(decoder.state_dict(), './model2/decoder_parameters_epoch%d.pkl' % (epoch))
        
    f.close()
    
    return loss


def test(decoder_a):
    decoder_a.eval()
    loss = 0
    for datas in test_loader:
        datas = datas.to(device)
        pred = decoder_a(datas)
        
#        print(pred)
        loss += criterion(pred, datas).item()
        
    return loss / (run2 * 2 * L ** 2)
#    return loss / run2


if __name__ == '__main__':
    training = 1
    load = 0
    if training:
        for epoch in range(1, 481):
            train(epoch)
            test_acc = test(decoder)
            print('Epoch: {:03d}, Test Acc: {:.10f}'.format(epoch, test_acc))
    
#    if load:
#        for i in range(6, 211, 6):
#            f = open('./test_loss_for_quantum.txt','a')
#            decoder_a = GNNI(Nc).to(device)
#            decoder_a.load_state_dict(torch.load('./model/decoder_parameters_epoch%d.pkl' % (i)))
#            loss = test(decoder_a)
#            print(loss)
#            f.write(' %.15f ' % (loss))
#            
#            f.close()
            
    if load:
        f = open('./test_loss_for_trained_model.txt','a')
        decoder_b = GNNI(Nc).to(device)
        decoder_b.load_state_dict(torch.load('./model/decoder_parameters_epoch132.pkl'))
        
        loss = test(decoder_b)
        print(loss)
        f.write(' %.15f ' % (loss))
        
        f.close()
#   