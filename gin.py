import os.path as osp
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity as cos
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINConv, global_add_pool
from torch_scatter import scatter_add, scatter_max
import gudhi as gd

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import sys

# DNN
class CNN(torch.nn.Module):
    def __init__(self, dim_out):
        super(CNN, self).__init__()
        self.dim_out = dim_out
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, kernel_size=2, stride=2), #channel of ZPI is 1
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(8, dim_out, kernel_size=2, stride=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.maxpool = torch.nn.MaxPool2d(5,5)

    def forward(self, PI):
        feature = self.features(PI)
        feature = self.maxpool(feature)
        feature = feature.view(-1, self.dim_out) #B, dim_out
        return feature

class Encoder(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers):
        super(Encoder, self).__init__()

        self.num_gc_layers = num_gc_layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_gc_layers):

            if i:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            else:
                nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
            conv = GINConv(nn)
            bn = torch.nn.BatchNorm1d(dim)

            self.convs.append(conv)
            self.bns.append(bn)

    def forward(self, x, edge_index, batch):
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(device)

        xs = []
        for i in range(self.num_gc_layers):
            x = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](x)
            xs.append(x)
            # if i == 2:
                # feature_map = x2

        xpool = [global_add_pool(x, batch) for x in xs]

        x = torch.cat(xpool, 1)

        return x, torch.cat(xs, 1)

    def get_embeddings(self, loader):

        device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:

                data = data[0]
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0],1)).to(device)
                x, _ = self.forward(x, edge_index, batch)

                ret.append(x.cpu().numpy())
                y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y

    def get_topo_embeddings(self, loader):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:

                data = data[0]
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0],1)).to(device)
                x, _ = self.forward(x, edge_index, batch)

                ret.append(x.cpu().numpy())
                y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y



class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        try:
            num_features = dataset.num_features
        except:
            num_features = 1
        dim = 32

        self.encoder = Encoder(num_features, dim)

        self.fc1 = Linear(dim*5, dim)
        self.fc2 = Linear(dim, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)

        x, _ = self.encoder(x, edge_index, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

def train(epoch):
    model.train()

    if epoch == 51:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.5 * param_group['lr']

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()

    return loss_all / len(train_dataset)

def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)

if __name__ == '__main__':

    for percentage in [ 1.]:
        for DS in [sys.argv[1]]:
            if 'REDDIT' in DS:
                epochs = 200
            else:
                epochs = 100
            path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', DS)
            accuracies = [[] for i in range(epochs)]
            dataset = TUDataset(path, name=DS) #.shuffle()
            num_graphs = len(dataset)
            print('Number of graphs', len(dataset))
            dataset = dataset[:int(num_graphs * percentage)]
            dataset = dataset.shuffle()

            kf = KFold(n_splits=10, shuffle=True, random_state=None)
            for train_index, test_index in kf.split(dataset):

                train_dataset = [dataset[int(i)] for i in list(train_index)]
                test_dataset = [dataset[int(i)] for i in list(test_index)]
                print('len(train_dataset)', len(train_dataset))
                print('len(test_dataset)', len(test_dataset))

                train_loader = DataLoader(train_dataset, batch_size=128)
                test_loader = DataLoader(test_dataset, batch_size=128)

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = Net().to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

                for epoch in range(1, epochs+1):
                    train_loss = train(epoch)
                    train_acc = test(train_loader)
                    test_acc = test(test_loader)
                    accuracies[epoch-1].append(test_acc)
                    tqdm.write('Epoch: {:03d}, Train Loss: {:.7f}, '
                          'Train Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, train_loss,
                                                                       train_acc, test_acc))
            tmp = np.mean(accuracies, axis=1)
            print(percentage, DS, np.argmax(tmp), np.max(tmp), np.std(accuracies[np.argmax(tmp)]))
            input()
