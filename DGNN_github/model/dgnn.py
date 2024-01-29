import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Parameter, Linear, Sequential, LogSoftmax


class DGNN(Module):
    def __init__(self, n, nclass, nfeat, nlayer, lambda_1, lambda_2, lambda_3, epsilon, dropout):
        super(DGNN, self).__init__()
        self.n = n
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.epsilon = epsilon
        self.nclass = nclass
        self.nfeat = nfeat
        self.nlayer = nlayer
        self.dropout = dropout
        self.w2 = Sequential(Linear(3 * nfeat, nfeat), LogSoftmax(dim=1))
        self.wy = Parameter(torch.FloatTensor(nfeat, nfeat), requires_grad=True)
        self.params2 = list(self.w2.parameters())
        self.params23 = [self.wy]
        self.laplacian1 = None
        self.laplacian2 = None
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.uniform_(self.wy, 0, 0.01)

    def forward(self, feat, adj, adj1):
        if self.laplacian1 is None and self.laplacian2 is None:
            n = adj.shape[0]
            indices = torch.Tensor([list(range(n)), list(range(n))])
            values = torch.FloatTensor([1.0] * n)
            eye = torch.sparse_coo_tensor(indices=indices, values=values, size=[n, n]).to(adj.device)
            self.laplacian1 = adj
            self.laplacian2 = adj1
        lap1 = self.laplacian1
        lap2 = self.laplacian2
        lap2 = lap2.to('cuda:0')
        lap2 = lap2.to_dense()
        y: Tensor = torch.rand(self.n, self.nfeat).to(adj.device)
        z1: Tensor = feat
        z2: Tensor = feat
        for i in range(self.nlayer):
            feat = F.dropout(feat, self.dropout, training=self.training)
            temp01 = torch.mm(y, self.wy)
            temp01 = torch.mm(temp01,self.wy.t())
            temp01 = torch.mm(temp01, y.t())
            temp02 = torch.mm(z1, self.wy)
            temp02 = torch.mm(temp02,self.wy.t())
            temp02 = torch.mm(temp02, z1.t())
            temp03 = torch.mm(z2, self.wy)
            temp03 = torch.mm(temp03, self.wy.t())
            temp03 = torch.mm(temp03, z2.t())
            temp11 = temp01 - self.epsilon * temp02 - (1-self.epsilon) * temp03
            temp04 = torch.mm(y, self.wy)
            temp04 = torch.mm(temp04, self.wy.t())
            temp05 = torch.mm(y, self.wy.t())
            temp05 = torch.mm(temp05, self.wy)
            temp12 = temp04 + temp05
            temp1 = torch.mm(temp11, temp12)
            temp1 = torch.sigmoid(temp1)
            y_n = feat - self.lambda_3 * temp1
            temp21 = self.epsilon * temp02 + (1-self.epsilon)*temp03 - temp01
            temp22 = torch.mm(z1, self.wy)
            temp22 = torch.mm(temp22,self.wy.t())
            temp23 = torch.mm(z1, self.wy.t())
            temp23 = torch.mm(temp23, self.wy)
            temp24 = temp22 + temp23
            temp2 = torch.mm(temp21, temp24)
            temp2 = torch.sigmoid(temp2)
            temp2 = self.epsilon * (self.lambda_3 / self.lambda_1) * temp2
            z1_n = torch.spmm(lap1, z1) - temp2
            temp31 = temp21
            temp32 = torch.mm(z2, self.wy)
            temp32 = torch.mm(temp32, self.wy.t())
            temp33 = torch.mm(z2, self.wy.t())
            temp33 = torch.mm(temp33, self.wy)
            temp34 = temp32 + temp33
            temp3 = torch.mm(temp31, temp34)
            temp3 = torch.sigmoid(temp3)
            temp3 =(1-self.epsilon) * (self.lambda_3 / self.lambda_2) * temp3
            z2_n = torch.spmm(lap2, z2) - temp3
            y = y_n
            z1 = z1_n
            z2 = z2_n
        y = F.normalize(y, p=2, dim=1)
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        p = torch.cat((y, z1, z2), dim=1)
        p = F.dropout(p, self.dropout, training=self.training)
        return self.w2(p)



