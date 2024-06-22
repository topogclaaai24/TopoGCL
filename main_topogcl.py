import os.path as osp
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from itertools import cycle
from torch_geometric.data import InMemoryDataset, download_url, Data
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from torch import Tensor
from aug import TUDataset_aug as TUDataset
from torch_geometric.data import DataLoader
import sys
import json
from torch import optim

from cortex_DIM.nn_modules.mi_networks import MIFCNet, MI1x1ConvNet
from losses import *
from gin import Encoder, CNN
from topogin import TopoEncoder
from evaluate_embedding import evaluate_embedding
from model import *
import networkx as nx
from arguments import arg_parse
from torch_geometric.transforms import Constant
import pdb
from tda_subfiltration import apply_graph_extended_persistence, persistence_images


def MyDataset(dataset, PIs, type = 'ori'):
    dataset1 = list()
    for i in range(len(dataset)):
        if type == 'ori':
            tmp_data = dataset[i][0]
        else:
            tmp_data = dataset[i][1]

        final_pi = PIs[i,:,:]
        x_topo = torch.FloatTensor(final_pi)
        data1 = Data(x=x_topo, y=tmp_data.y)
        dataset1.append(data1)
    return dataset1


class simclr(nn.Module):
  def __init__(self, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
    super(simclr, self).__init__()

    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.prior = args.prior

    self.embedding_dim = mi_units = hidden_dim * num_gc_layers
    self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)
    self.topoencoder = TopoEncoder(dataset_num_features, hidden_dim * num_gc_layers, num_gc_layers)
    self.cnn = CNN(dim_out = self.embedding_dim)

    self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True), nn.Linear(self.embedding_dim, self.embedding_dim))

    self.init_emb()

  def init_emb(self):
    initrange = -1.5 / self.embedding_dim
    for m in self.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

  def forward(self, x, topo_x, edge_index, batch, num_graphs, topo=False):
      if topo:
          resolution_size = 50
          topo_x = topo_x.view(-1, 1, resolution_size, resolution_size)
          topo_y = self.topoencoder(topo_x)
          topo_y = self.proj_head(topo_y)
          final_y = topo_y
      else:
          if x is None:
              x = torch.ones(batch.shape[0]).to(device)

          y, M = self.encoder(x, edge_index, batch)

          y = self.proj_head(y)

          final_y = y

      return final_y

  def loss_cal(self, x, x_aug, x_topo, x_topo_aug):

      T = 0.2
      batch_size, _ = x.size()
      x_abs = x.norm(dim=1)
      x_aug_abs = x_aug.norm(dim=1)
      sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
      sim_matrix = torch.exp(sim_matrix / T)
      pos_sim = sim_matrix[range(batch_size), range(batch_size)]
      loss1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)

      x_topo_abs = x_topo.norm(dim=1)
      x_topo_aug_abs = x_topo_aug.norm(dim=1)
      sim_matrix_topo = torch.einsum('ik,jk->ij', x_topo, x_topo_aug) / torch.einsum('i,j->ij', x_topo_abs,
                                                                                     x_topo_aug_abs)
      sim_matrix_topo = torch.exp(sim_matrix_topo / T)
      pos_sim_topo = sim_matrix_topo[range(batch_size), range(batch_size)]
      loss2 = pos_sim_topo / (sim_matrix_topo.sum(dim=1) - pos_sim_topo)

      loss = 1. * loss1 + 0.1 * loss2

      loss = - torch.log(loss).mean()

      return loss


import random
def setup_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    
    args = arg_parse()
    setup_seed(args.seed)

    accuracies = {'val':[], 'test':[]}
    epochs = 200
    log_interval = 5
    batch_size = 128
    lr = args.lr
    DS = 'PTC_MR'
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS)

    dataset = TUDataset(path, name=DS, aug=args.aug).shuffle()
    dataset_eval = TUDataset(path, name=DS, aug='none').shuffle()
    resolution_size = 50

    #'''
    print("seed is:", args.seed)
    ori_PIs = np.load(DS + '_topo_data/' + 'seed=' + str(args.seed) + '/' + DS +'_ORI_BetEPIs_seed_' + str(args.seed) + '.npz', allow_pickle=True)['arr_0']
    aug_PIs = np.load(DS + '_topo_data/' + 'seed=' + str(args.seed) + '/' + DS + '_AUG_BetEPIs_seed_' + str(args.seed) + '.npz', allow_pickle=True)['arr_0']


    ori_dataset_topo = MyDataset(dataset = dataset, PIs= ori_PIs, type = 'ori')
    aug_dataset_topo = MyDataset(dataset=dataset, PIs= aug_PIs, type='aug')

    data_pis = []
    for kk in range(len(dataset)):
        print("kk is:", kk)
        tmp_ori_data = dataset[kk][0]
        tmp_aug_data = dataset[kk][1]
        tmp_ori_eval_data = dataset_eval[kk][0]

        edge_index = ((tmp_aug_data.edge_index).numpy()).transpose()
        net = nx.from_edgelist(edge_index)

        adj = nx.adjacency_matrix(net).toarray()
        betweenness_scores = np.array(list(nx.betweenness_centrality(net).values()))
        nodes_degree = np.sum(adj, axis=1)
        final_dgm = apply_graph_extended_persistence(A=adj, filtration_val=betweenness_scores)
        final_pi = persistence_images(final_dgm, resolution=[50, 50])
        data_pis.append(final_pi)

    try:
        dataset_num_features = dataset.get_num_feature()
    except:
        dataset_num_features = 1

    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers = 16, pin_memory = True)
    dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size, num_workers = 16, pin_memory = True)

    ori_dataloader_topo = DataLoader(ori_dataset_topo, batch_size= batch_size, num_workers = 16, pin_memory = True)
    aug_dataloader_topo = DataLoader(aug_dataset_topo, batch_size=batch_size, num_workers = 16, pin_memory = True)


    model = simclr(args.hidden_dim, args.num_gc_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print('================')
    print('lr: {}'.format(lr))
    print('num_features: {}'.format(dataset_num_features))
    print('hidden_dim: {}'.format(args.hidden_dim))
    print('num_gc_layers: {}'.format(args.num_gc_layers))
    print('================')

    model.eval()
    emb, y = model.encoder.get_embeddings(dataloader_eval)

    for epoch in range(1, epochs+1):
        loss_all = 0
        model.train()
        for i, (data, data_topo, data_aug_topo) in enumerate(zip(cycle(dataloader), ori_dataloader_topo, aug_dataloader_topo)):
            data, data_aug = data
            data_topo = data_topo.to(device, non_blocking=True)
            data_aug_topo = data_aug_topo.to(device, non_blocking=True)

            optimizer.zero_grad()

            node_num, _ = data.x.size()
            data = data.to(device, non_blocking=True)
            x = model(data.x, data_topo.x, data.edge_index, data.batch, data.num_graphs, topo=False)
            x_topo = model(data.x, data_topo.x, data.edge_index, data.batch, data.num_graphs, topo=True)

            if args.aug == 'dnodes' or args.aug == 'subgraph' or args.aug == 'random2' or args.aug == 'random3' or args.aug == 'random4':
                edge_idx = data_aug.edge_index.numpy()
                _, edge_num = edge_idx.shape
                idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]

                node_num_aug = len(idx_not_missing)
                data_aug.x = data_aug.x[idx_not_missing]

                data_aug.batch = data.batch[idx_not_missing]
                idx_dict = {idx_not_missing[n]:n for n in range(node_num_aug)}
                edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_num) if not edge_idx[0, n] == edge_idx[1, n]]
                data_aug.edge_index = torch.tensor(edge_idx).transpose_(0, 1)

            data_aug = data_aug.to(device, non_blocking=True)


            x_aug = model(data_aug.x, data_aug_topo.x, data_aug.edge_index, data_aug.batch, data_aug.num_graphs, topo=False)

            x_topo_aug = model(data_aug.x, data_aug_topo.x, data_aug.edge_index, data_aug.batch, data_aug.num_graphs, topo=True)

            loss = model.loss_cal(x, x_aug, x_topo, x_topo_aug)


            loss_all += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()
        print('Epoch {}, Loss {}'.format(epoch, loss_all / len(dataloader)))

        if epoch % log_interval == 0:
            model.eval()
            emb, y = model.encoder.get_embeddings(dataloader_eval)
            acc_val, acc = evaluate_embedding(emb, y)
            accuracies['val'].append(acc_val)
            accuracies['test'].append(acc)
            accuracies['test'].append(acc)
            print(accuracies['val'][-1], accuracies['test'][-1])


    tpe  = ('local' if args.local else '') + ('prior' if args.prior else '')
    with open('logs/log_' + args.DS + '_' + args.aug, 'a+') as f:
        s = json.dumps(accuracies)
        f.write('{},{},{},{},{},{},{}\n'.format(args.DS, tpe, args.num_gc_layers, epochs, log_interval, lr, s))
