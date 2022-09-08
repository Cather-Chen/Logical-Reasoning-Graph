
from layers import *
from torch.nn.parameter import Parameter
from functools import reduce
import torch



class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.normalize = False
        self.attention = False
        
        self.gc1 = GraphConvolution(nfeat, nhid, bias=False)
        self.gc2 = GraphConvolution(nhid, nclass, bias=False)
            
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        if self.normalize:
            x = F.normalize(x, p=2, dim=1)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        if self.normalize:
            x = F.normalize(x, p=2, dim=1)
        x = F.relu(x)
        return F.log_softmax(x, dim=1)


class HGAT(nn.Module):
    '''
    input: feature_list, adj_list
    return:list(len = n_type) of tensor(bsz * 4, n_nodes, dim2)
    '''
    def __init__(self, nfeat_list, nhid, out_dim, dropout):
        super(HGAT, self).__init__()
        self.para_init()
        self.attention = True
        self.lower_attention = True  #是否需要交叉type的attention
        self.embedding = False  #是否需要先线性化映成同样维度的embedding
        self.write_emb = False
        dim_1st = nhid
        dim_2nd = out_dim
        self.norm1 = torch.nn.LayerNorm(dim_1st,eps=1e-5,elementwise_affine= True)
        self.norm2 = torch.nn.LayerNorm(dim_2nd,eps=1e-5,elementwise_affine= True)
        if self.write_emb:
            self.emb = None

        self.nonlinear = F.elu_

        self.ntype = len(nfeat_list)   #nfeat_list=[d1,d2,d3] 表示三种type嵌入的维度
        if self.embedding:
            self.mlp = nn.ModuleList()
            n_in = [nhid for _ in range(self.ntype)]
            for t in range(self.ntype):
                self.mlp.append(MLP(nfeat_list[t], n_in[t]))
        else:
            n_in = nfeat_list

        # dim_1st = 1000
        # dim_2nd = nhid

        
        self.gc2 = nn.ModuleList()
        if not self.lower_attention:
            self.gc1 = nn.ModuleList()
            for t in range(self.ntype):
                self.gc1.append(GraphConvolution(n_in[t], dim_1st, bias=False) )
                self.bias1 = Parameter(torch.FloatTensor(dim_1st))
                stdv = 1. / math.sqrt(dim_1st)
                self.bias1.data.uniform_(-stdv, stdv)
        else:
            self.gc1 = GraphAttentionConvolution(n_in, dim_1st, gamma=0.1)
        self.gc2.append(GraphConvolution(dim_1st, dim_2nd, bias=True) )
        # self.gc2 = GraphAttentionConvolution([dim_1st] * self.ntype, dim_2nd)

        if self.attention:
            self.at1 = nn.ModuleList()
            self.at2 = nn.ModuleList()
            for t in range(self.ntype):
                self.at1.append(SelfAttention(dim_1st, t, 50) )  #(in_features, idx, hidden_dim)
                self.at2.append(SelfAttention(dim_2nd, t, 50) )
           
        # self.outlayer = torch.nn.Linear(dim_2nd, nclass)

        self.dropout = dropout

    def para_init(self):
        self.attention = True
        self.embedding = False
        self.lower_attention = False
        self.write_emb = False

    def forward(self, x_list, adj_list):
        if self.embedding:
            x0 = [None for _ in range(self.ntype)]
            for t in range(self.ntype):
                x0[t] = self.mlp[t](x_list[t])
        else:
            x0 = x_list
        
        if not self.lower_attention:
            x1 = [None for _ in range(self.ntype)]
            # 第一层gat，与第一层后的dropout
            for t1 in range(self.ntype):
                x_t1 = []
                for t2 in range(self.ntype):
                    if adj_list[t1][t2] is None:
                        continue
                    idx = t2
                    x_t1.append(self.gc1[idx](x0[t2], adj_list[t1][t2]) + self.bias1 )
                if self.attention:
                    x_t1, weights = self.at1[t1](torch.stack(x_t1, dim=1) )
                else:
                    x_t1 = reduce(torch.add, x_t1)

                x_t1 = self.nonlinear(x_t1)
                x_t1 = F.dropout(x_t1, self.dropout, training=self.training)
                x1[t1] = x_t1
        else:
            x1 = [None for _ in range(self.ntype)]
            x1_in = self.gc1(x0, adj_list)  # list（len=type_num）of list(len=type_num) of tensor： size=（bsz*4， n_nodes，dim1）
            for t1 in range(len(x1_in)):
                x_t1 = x1_in[t1]  #list of tensor:size = (bsz*4,n_nodes, dim1)
                if self.attention:
                    x_t1, weights = self.at1[t1](torch.stack(x_t1, dim=1)) #stack: (bsz*4, 2,n_nodes,dim1)
                    #x_t1: tensor(bsz * 4, n_nodes, dim1)
                    #weights: tensor(bsz * 4, n_nodes, 2, 1)
                    # if t1 == 0:
                        # self.f.write('{0}\t{1}\t{2}\n'.format(weights[0][0].item(), weights[0][1].item(), weights[0][2].item()))
                else:
                    x_t1 = reduce(torch.add, x_t1)
                # x_t1 = self.norm1(x_t1)
                x_t1 = F.dropout(x_t1, self.dropout, training=self.training)
                x_t1 = self.nonlinear(x_t1)
                x1[t1] = x_t1
        if self.write_emb:
            self.emb = x1[0]        
        
        x2 = [None for _ in range(self.ntype)]
        # 第二层gcn，与第二层后的softmax
        for t1 in range(self.ntype):
            x_t1 = []
            for t2 in range(self.ntype):
                if adj_list[t1][t2] is None:
                    continue
                x_t1.append(self.gc2[0](x1[t2], adj_list[t1][t2]))  #append(tensor(bsz * 4, n_nodes, dim2))
            if self.attention:
                x_t1, weights = self.at2[t1](torch.stack(x_t1, dim=1))  #stack: (bsz*4, 2,n_nodes,dim2)  --> tensor(bsz * 4, n_nodes, dim2)
            else:
                x_t1 = reduce(torch.add, x_t1)

            x_t1 = F.dropout(x_t1, self.dropout, training=self.training)
            x_t1 = self.nonlinear(x_t1)
            x2[t1] = x_t1    #tensor(bsz * 4, n_nodes, dim2)
        return x2


        
    def inference(self, x_list, adj_list, adj_all = None):
        return self.forward(x_list, adj_list, adj_all)

        
class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.6, alpha=0.2, nheads=8):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, features, adjs):
        bsz = features.size()[0]
        device = features.device
        output = torch.zeros_like(features).to(device)
        for bs in range(bsz):
            x = features[bs,:,:]
            adj = adjs[bs,:,:]
            x = F.dropout(x, self.dropout, training=self.training)
            x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
            x = F.dropout(x, self.dropout, training=self.training)
            x = F.elu(self.out_att(x, adj))
            output[bs] = x

        return output




