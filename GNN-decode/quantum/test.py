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
            out = scatter_(self.aggr, out, edge_index[j], dim_size=size[i])[edge_index[j]] - out
        else:
            out = scatter_(self.aggr, out, edge_index[j], dim_size=size[i])[edge_index[j]] - out
            out = out + extra[edge_index[j]]
        
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
L = 6
P2 = [0.1]
H = torch.from_numpy(error_generate.generate_PCM(2 * L * L - 2, L)).t() #64, 30
h_prep = error_generate.H_Prep(H.t())
H_prep = torch.from_numpy(h_prep.get_H_Prep())
BATCH_SIZE = 1
Nc = 1
run2 = 1
dataset2 = error_generate.gen_syn(P2, L, H, run2)
test_dataset = CustomDataset(H, dataset2)
rows, cols = H.size(0), H.size(1)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=False)


def init_weights(m):
    if type(m) == torch.nn.Linear:
#        torch.nn.init.uniform_(m.weight, a=0.1138, b=0)
        torch.nn.init.constant_(m.weight, 0.0224)
        m.bias.data.fill_(1e-3)
        

class GatedGraphConv(MessagePassing):
    def __init__(self, flow, aggr='add', bias=True):
        super(GatedGraphConv, self).__init__(aggr, flow)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(2, 16).double(),
                               torch.nn.Softplus(),
                               torch.nn.Linear(16, 16).double(),
                               torch.nn.Softplus(),
                               torch.nn.Linear(16, 1).double())
        self.mlp.apply(init_weights)
        
    def forward(self, edge_index):
        mes = self.mlp(edge_index.double().t())
        mes2 = self.mlp(edge_index[0:2, :].double().t() ) #/ edge_index.max()
#        print((mes.sum() / mes.numel()).item())
#        print((mes2.sum() / mes2.numel()).item())
        print(mes.t())
        print(mes2.t())
        
        return mes

    
class GNNI(torch.nn.Module):
    def __init__(self, Nc):
        super(GNNI, self).__init__()
        self.ggc = GatedGraphConv("source_to_target")
        self.Nc = Nc
    
    def forward(self, data):
        edge_index = torch.cat([data.edge_index[0].unsqueeze(0), data.edge_index[1].unsqueeze(0).add(rows)], dim=0)
        
        m = self.ggc(edge_index)
        
        return 0


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test(decoder_a):
    decoder_a.eval()
    
    for datas in test_loader:
        datas = datas.to(device)
        pred = decoder_a(datas)
        
    return 0

if __name__ == '__main__':
    decoder = GNNI(Nc).to(device)
    
    loss = test(decoder)
    
    
    
