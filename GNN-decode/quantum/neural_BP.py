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
import torch_scatter
from torch_scatter import scatter_add
#import decoder_v2_4
'''
BP free decoder, with phase one & two modified to mlp
'''


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


L = 4
P1 = [0.01,0.02,0.03,0.04,0.05,0.06]
P2 = [0.01]
H = torch.from_numpy(error_generate.generate_PCM(2 * L * L - 2, L)[0]).t() #64, 30
h_prep = error_generate.H_Prep(H.t())
H_prep = torch.from_numpy(h_prep.get_H_Prep())
BATCH_SIZE = 128
lr = 3e-4
Nc = 15
run1 = 81920
run2 = 2048
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
'''
generating edge features
'''
feat = H.clone()
nb_digits = 4

for j in range(cols):
    tmp = feat[:, j].clone().to_sparse()
    i = tmp._indices()
    v = torch.tensor([0.2,1,2,3]).double()
    feat[:, j] = torch.sparse.FloatTensor(i, v, feat[:, j].size()).to_dense()
  
feat = feat.to_sparse()._values().unsqueeze(1)
feat_onehot = torch.Tensor(int(H.sum()), nb_digits).double()
feat_onehot.zero_()
feat_onehot.scatter_(1, feat.to(dtype=torch.long), 1)
feat_onehot = feat_onehot.repeat(BATCH_SIZE, 1).cuda()


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
#        torch.nn.init.constant_(m.weight, 0.01)
        m.bias.data.fill_(0)
        
        
def init_weights_2(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
#        torch.nn.init.constant_(m.weight, 0.01)
        m.bias.data.fill_(0)


def a_p(grad):
    a = torch.where(abs(grad) < 1e-1, torch.ones(grad.size()).cuda(), torch.zeros(grad.size()).cuda())
    print(abs(grad).min().item(), abs(grad).max().item(), a.sum().item() / grad.numel())
    

class GraphConv(MessagePassing):
    def __init__(self, flow, aggr='add', bias=True):
        super(GraphConv, self).__init__(aggr, flow)
        
        self.flow = flow
        
        self.W = torch.nn.Parameter(Variable(torch.ones((nb_digits, 1)).double()))
        self.W_p = torch.nn.Parameter(Variable(torch.ones((nb_digits, 1)).double()))

    def forward(self, m, edge_index, x, prev=None):
        x = x if x.dim() == 2 else x.unsqueeze(-1)
        
        if self.flow == 'source_to_target':
            m = torch.matmul(m.mul(feat_onehot), self.W)
        
        m = self.propagate(edge_index=edge_index, size=((rows+cols) * BATCH_SIZE, (rows+cols) * BATCH_SIZE), x=m, extra=x)
        
        return m
    
    def update(self, aggr_out):
        if self.flow == 'source_to_target':
            aggr_out[:, 1] = torch.matmul(aggr_out[:, 1].clone().unsqueeze(1).mul(feat_onehot), self.W_p).squeeze(1)
            
            return aggr_out[:, 0].clone().unsqueeze(1) + aggr_out[:, 1].clone().unsqueeze(1)
        else:
            return aggr_out
    
    
class GNNI(torch.nn.Module):
    def __init__(self, Nc):
        super(GNNI, self).__init__()
        
        self.Nc = Nc
        self.layers = self._make_layer()
        self.W = torch.nn.Parameter(Variable(torch.ones((nb_digits, 1)).double()))
        self.W_p = torch.nn.Parameter(Variable(torch.ones((nb_digits, 1))*0.5).double())
        self.alpha = torch.nn.Parameter(Variable(torch.Tensor([[6]]).double()))
        self.beta = torch.nn.Parameter(Variable(torch.Tensor([[-6]]).double()))
    
    def _make_layer(self):
        layers = []
        
        for _ in range(self.Nc):
            layers.append(GraphConv("source_to_target"))
            layers.append(GraphConv("target_to_source"))
            
        return torch.nn.Sequential(*layers)
    
    def forward(self, data):
        '''
        this part need to sum up the message and return
        '''
        x = data.x
        edge_index = torch.cat([data.edge_index[0].clone().unsqueeze(0), data.edge_index[1].clone().unsqueeze(0).add(rows)], dim=0)
        m = Variable(torch.zeros((edge_index.size()[1], 1), dtype = torch.float64).cuda(), requires_grad=False)
        
        size=((rows+cols) * BATCH_SIZE, (rows+cols) * BATCH_SIZE)
        idx = torch.LongTensor([x for x in range(rows)]).cuda()
        
#        feat_p = torch.Tensor([[x for x in range(rows)]]).t()
#        feat_p_onehot = torch.Tensor(rows, rows).double()
#        feat_p_onehot.zero_()
#        feat_p_onehot.scatter_(1, feat_p.to(dtype=torch.long), 1)
#        feat_p_onehot = feat_p_onehot.repeat(BATCH_SIZE, 1).cuda()
        
        for i in range(rows+cols, len(x), rows+cols):
            idx = torch.cat([idx, torch.LongTensor([x for x in range(i, i+rows)]).cuda()], dim=0)
        
        for i in range(0, len(self.layers), 2):
            m_p = m.clone()
            m = self.layers[i](m, edge_index, x)
            m = torch.matmul(self.layers[i+1](m, edge_index, x), torch.sigmoid(self.alpha)) + \
            torch.matmul(m_p, torch.sigmoid(self.beta))
#            a = scatter_('add', self.mlp(m), edge_index[0], dim_size=size[0])[idx].clone() + x[idx]
#            print(torch.sigmoid(-1 * a).t())
        m = torch.matmul(m.mul(feat_onehot), self.W)
        prior = torch.matmul(x[edge_index[0]].mul(feat_onehot), self.W_p)
        res = scatter_('add', m, edge_index[0], dim_size=size[0])[idx].clone() + \
        scatter_('add', prior, edge_index[0], dim_size=size[0])[idx].clone()
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
    
    
class WeightClipper(object):
    def __init__(self, frequency=5):
        self.frequency = frequency
    
    def __call__(self, module):
        if hasattr(module, 'W'):
            w = module.W.data
            w = w.clamp(1e-10, 1e10)
            module.W.data = w
            
        if hasattr(module, 'W_p'):
            w = module.W_p.data
            w = w.clamp(1e-10, 1e10)
            module.W_p.data = w


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
decoder = GNNI(Nc).to(device)
'''
apply weight clipper
'''
clipper = WeightClipper()
decoder.apply(clipper)

#decoder.load_state_dict(torch.load('./neural_BP/decoder_parameters_epoch20.pkl'))
optimizer = torch.optim.Adam(decoder.parameters(), lr, weight_decay=0)
criterion = LossFunc(H, H_prep)
'''
load pretrained model
'''
#pretrained = decoder_v2_4.GNNI(Nc).to(device)
#pretrained.load_state_dict(torch.load('./new_model/decoder_parameters_epoch67.pkl'))
#decoder_dict = decoder.state_dict() 
#pretrained_dict = {k: v for k, v in pretrained.state_dict().items() if k in decoder_dict}
#decoder_dict.update(pretrained_dict)
#decoder.load_state_dict(decoder_dict)


def train(epoch):
    decoder.train()
    f = open('./training_loss_for_quantum.txt','a')
    
    for datas in train_loader:
        datas = datas.to(device)
        optimizer.zero_grad()
        loss = criterion(decoder(datas), datas)
        loss.backward()
        optimizer.step()
        
    if epoch % 1 == 0:
        f.write(' %.15f ' % (loss.item()))
        torch.save(decoder.state_dict(), './neural_BP/decoder_parameters_epoch%d.pkl' % (epoch))
        
    f.close()
    
    return loss


def test(decoder_a):
    decoder_a.eval()
    loss = 0
    for datas in test_loader:
        datas = datas.to(device)
        pred = decoder_a(datas)
        
        loss += criterion(pred, datas).item()
        
    return loss / (run2 * L ** 2)
#    return loss / run2


if __name__ == '__main__':
    training = 1
    load = 1 - training
    if training:
        for epoch in range(1, 1201):
            train(epoch)
            test_acc = test(decoder)
            print('Epoch: {:03d}, Test Acc: {:.10f}'.format(epoch, test_acc))
    
            
    if load:
        decoder_b = GNNI(Nc).to(device)
        decoder_b.load_state_dict(torch.load('./neural_BP/decoder_parameters_epoch407.pkl'))
        
        loss = test(decoder_b)
        print(loss)
