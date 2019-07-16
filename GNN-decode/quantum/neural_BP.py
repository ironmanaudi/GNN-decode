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
'''
adding graph-tolology-awareness attribute explicitly by attatching learnable weight to each message to build a nueral BP covering 
algorithm
'''

torch.autograd.set_detect_anomaly(True)
torch.set_printoptions(threshold=100000)
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

#                    tmp = torch.index_select(tmp, 0, edge_index[idx]) # function of this step
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
#            out = torch.clamp(out, -10, 10)
            out = torch.tanh(out / 2)
            Coeff = torch.where(out < 0, torch.ones(out.size(), dtype=torch.float64).cuda(), \
                                torch.zeros(out.size(), dtype=torch.float64).cuda())
            out = abs(out)
            out = torch.clamp(out, 1e-20, 1e10)
            out = torch.log(out)
            out = scatter_(self.aggr, out, edge_index[j], dim_size=size[i])[edge_index[j]] - out
            
            Coeff = scatter_(self.aggr, Coeff, edge_index[j], dim_size=size[i])[edge_index[j]] - Coeff
            Coeff = torch.cos(math.pi * (Coeff + (1 - extra[edge_index[j]]) / 2))
            out = torch.exp(out).mul(Coeff)
            out = torch.clamp(out, -1+1e-15, 1-1e-15)
            out = torch.log(1 + out) - torch.log(1 - out)
        else:
            out = scatter_(self.aggr, out, edge_index[j], dim_size=size[i])[edge_index[j]] - out
        
        if self.flow == 'source_to_target':
#            out = out + extra[edge_index[j]]
            out = torch.cat([out, extra[edge_index[j]]], dim=1)
#        else:
#            out = torch.cat([out, extra[edge_index[j]]], dim=1)
            
        out = self.update(out, *update_args)

        return out


    def message(self, x):  # pragma: no cover
        
        return x


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


L = 5
P1 = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
P2 = [0.01]
H = torch.from_numpy(error_generate.generate_PCM(2 * L * L - 2, L)).t() #64, 30
h_prep = error_generate.H_Prep(H.t())
H_prep = torch.from_numpy(h_prep.get_H_Prep())
BATCH_SIZE = 128
lr = 3e-4
Nc = 15
run1 = 2048#40960
run2 = 256#2048
index = torch.LongTensor([1,0])
adj = H.to_sparse()
edge_info = torch.cat([adj._indices()[0].unsqueeze(0), \
                         adj._indices()[1].unsqueeze(0).add(H.size()[0])], dim=0).repeat(1, BATCH_SIZE).cuda()
dataset1 = error_generate.gen_syn(P1, L, H, run1)
dataset2 = error_generate.gen_syn(P2, L, H, run2)
train_dataset = CustomDataset(H, dataset1)
test_dataset = CustomDataset(H, dataset2)
rows, cols = H.size(0), H.size(1)
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=False)
logical, stab = h_prep.get_logical(H_prep)
logical, stab = logical.cuda(), stab.cuda()


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        m.bias.data.fill_(0)
        
        
def init_weights_2(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        m.bias.data.fill_(0)
        

def a_p(grad):
    a = torch.where(abs(grad) < 1e-1, torch.ones(grad.size()).cuda(), torch.zeros(grad.size()).cuda())
    print(abs(grad).min().item(), abs(grad).max().item(), a.sum().item() / grad.numel())
    

class GraphConv(MessagePassing):
    def __init__(self, flow, aggr='add', bias=True):
        super(GraphConv, self).__init__(aggr, flow)
        
        self.flow = flow
        if self.flow == 'source_to_target':
            self.mlp = torch.nn.Sequential(#torch.nn.BatchNorm1d(3).double(),
                       torch.nn.Linear(1, 256).double(),
                       torch.nn.Softplus(),
                       torch.nn.Linear(256, 1).double())
            self.mlp.apply(init_weights)
            self.mlp1 = torch.nn.Sequential(#torch.nn.BatchNorm1d(3).double(),
                       torch.nn.Linear(2, 256).double(),
                       torch.nn.Softplus(),
                       torch.nn.Linear(256, 1).double())
            self.mlp1.apply(init_weights)
        
    def forward(self, m, edge_index, x):
        x = x if x.dim() == 2 else x.unsqueeze(-1)
        
        if self.flow == 'source_to_target':
            m = self.mlp1(edge_info.t().double()).mul(m)
            
        mes = self.propagate(edge_index=edge_index, size=((rows+cols) * BATCH_SIZE, (rows+cols) * BATCH_SIZE), x=m, extra=x)
        
        return mes
            
    def update(self, aggr_out):
        if self.flow == 'source_to_target':
            w = self.mlp(edge_info[0].unsqueeze(0).t().double())
            return aggr_out[:, 1].clone().unsqueeze(1).mul(w) + aggr_out[:, 0].clone().unsqueeze(1)
        else:
            return aggr_out
    
    
class GNNI(torch.nn.Module):
    def __init__(self, Nc):
        super(GNNI, self).__init__()
        
        self.Nc = Nc
        self.ggc1 = GraphConv("source_to_target")
        self.ggc2 = GraphConv("target_to_source")
        self.mlp = torch.nn.Sequential(torch.nn.Linear(3, 256).double(),
                       torch.nn.Softplus(),
                       torch.nn.Linear(256, 1).double())
        self.mlp1 = torch.nn.Sequential(torch.nn.Linear(1, 256).double(),
                       torch.nn.Softplus(),
                       torch.nn.Linear(256, 1).double())
#        self.mlp.apply(init_weights_2)
    
    def forward(self, data):
        '''
        this part need to sum up the message and return
        '''
        x = data.x
        edge_index = torch.cat([data.edge_index[0].clone().unsqueeze(0), data.edge_index[1].clone().unsqueeze(0).add(rows)], dim=0)
        m = Variable(torch.zeros((edge_index.size()[1], 1), dtype = torch.float64).cuda(), requires_grad=False)
        
        for i in range(self.Nc):
            m_p = m.clone()
            m = self.ggc1(m, edge_index, x)
            m = self.ggc2(m, edge_index, x) + m_p
        
        m = self.mlp(torch.cat([m, edge_info.t().double()], dim=1))
        w = self.mlp1(torch.Tensor([x for x in range(rows)]).repeat(1, BATCH_SIZE).double().t().cuda())
        
        size=((rows+cols) * BATCH_SIZE, (rows+cols) * BATCH_SIZE)
        idx = torch.LongTensor([x for x in range(rows)]).cuda()
        
        for i in range(rows+cols, len(x), rows+cols):
            idx = torch.cat([idx, torch.LongTensor([x for x in range(i, i+rows)]).cuda()], dim=0)
        
        res = scatter_('add', m, edge_index[0], dim_size=size[0])[idx].clone() + x[idx].mul(w)
        res = torch.sigmoid(-1 * res)
        
        return res


class LossFunc(torch.nn.Module):
    def __init__(self, H, H_prep):
        super(LossFunc, self).__init__()
        
        self.a, self.b = max(H.size()), min(H.size())
        self.H_prep = H_prep.cuda()
        
    def forward(self, pred, datas):
#        print(pred.t())
#        print(datas.y.t())
        tmp = datas.y[0 : self.a].clone()
        res = pred[0 : self.a].clone()
        
        for i in range(self.a, len(datas.y), self.a):
            tmp = torch.cat([tmp, datas.y[i : i+self.a].clone()], dim=1)
            res = torch.cat([res, pred[i : i+self.a]], dim=1)
            
        loss = abs(torch.sin(torch.matmul(H.t().cuda(), tmp + res) * math.pi / 2)).sum() + \
            abs(torch.sin(torch.matmul(logical, tmp + res) * math.pi / 2)).sum()
        
        return loss
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
decoder = GNNI(Nc).to(device)
#decoder.load_state_dict(torch.load('./model1_2/decoder_parameters_epoch1.pkl'))
optimizer = torch.optim.Adam(decoder.parameters(), lr, weight_decay=1e-9)
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
        
    if epoch % 1 == 0:
        f.write(' %.15f ' % (loss.item()))
        torch.save(decoder.state_dict(), './model1_2/decoder_parameters_epoch%d.pkl' % (epoch))
        
    f.close()
    
    return loss


def test(decoder_a):
    decoder_a.eval()
    loss = 0
    for datas in test_loader:
        datas = datas.to(device)
        pred = decoder_a(datas)
        
        loss += criterion(pred, datas).item()
        
    return loss / (run2 * 2 * L ** 2)
#    return loss / run2


if __name__ == '__main__':
    training = 1
    load = 1 - training
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
        decoder_b.load_state_dict(torch.load('./model1_2/decoder_parameters_epoch52.pkl'))
        
        loss = test(decoder_b)
        print(loss)
        f.write(' %.15f ' % (loss))
        
        f.close()