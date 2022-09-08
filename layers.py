import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch import nn
import allennlp as util
class GraphConvolution(Module):
    '''
    from in_features dim → out_features dim
    '''
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):  #初始化参数
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def normalize_adj_torch(self,adj):
        bsz = adj.size(0)
        adj_norm = torch.zeros_like(adj).to(adj.device)
        for bs in range(bsz):
            mx = adj[bs]
            rowsum = mx.sum(1)  # 每行的数加在一起
            r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()  # 输出rowsum ** -1/2
            r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.  # 溢出部分赋值为0
            r_mat_inv_sqrt = torch.diag(r_inv_sqrt)  # 对角化
            mx = torch.matmul(r_mat_inv_sqrt,mx)
            mx = torch.transpose(mx, 0, 1)  # 转置
            mx = torch.matmul(mx, r_mat_inv_sqrt)
            mx = torch.transpose(mx,0,1)
        adj_norm[bs] = mx
        return adj_norm


    def forward(self, inputs, adj, global_W = None): #input:(bsz, n_nodes,embsz),  adj:(bsz,n_nodes,m_nodes)
        # if adj == 0:
        #     return torch.zeros(adj.shape[0], self.out_features, device=inputs.device)
        adj = self.normalize_adj_torch(adj)
        support = torch.matmul(inputs, self.weight)  # (bsz*4,n_nodes, embsz) * (embsz, outsz)
        if global_W is not None:
            support = torch.matmul(support, global_W)
        output = torch.matmul(adj, support)# (bsz*4,n,n) (bsz*4,n_nodes, outsz)
        if self.bias is not None:
            return output + self.bias
        else:
            return output   #(bsz*4, n_nodes, outfeature_dim)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class SelfAttention_ori(Module):
    """
    SelfAttention
        return outputs, weights
    """
    def __init__(self, in_features):
        super(SelfAttention, self).__init__()
        self.a = Parameter(torch.FloatTensor(2 * in_features, 1))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.a.size(1))
        self.a.data.uniform_(-stdv, stdv)

    def forward(self, inputs):
        x = inputs.transpose(0, 1)
        self.n = x.size()[0]
        x = torch.cat([x, torch.stack([x] * self.n, dim=0)], dim=2)
        U = torch.matmul(x, self.a).transpose(0, 1)
        # 非线性激活
        U = F.leaky_relu(U)
        weights = F.softmax(U, dim=1)
        outputs = torch.matmul(weights.transpose(1, 2), inputs).squeeze(1)
        return outputs, weights




class SelfAttention(Module):
    """
        input：tensor (bsz*4,2,n_nodes,in_features)
        return:
            outputs:tensor (bsz*4,n_nodes, in_features)
            weight:tensor (bsz*4,n_nodes,2,1)
    """
    def __init__(self, in_features, idx, hidden_dim):
        super(SelfAttention, self).__init__()
        self.idx = idx
        self.linear = torch.nn.Linear(in_features, hidden_dim)
        self.a = Parameter(torch.FloatTensor(2 * hidden_dim, 1))
        self.reset_parameters()
        self.norm = torch.nn.LayerNorm(1,eps=1e-5,elementwise_affine=True)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.a.size(1))
        self.a.data.uniform_(-stdv, stdv)

    def forward(self, inputs):
        # inputs size:  bsz,2,node_num,in_features
        x = self.linear(inputs)  # x: (bsz*4,2,n_nodes,hidden_dim)
        self.n = x.size()[1]
        x = torch.cat([x, torch.stack([x[:,self.idx,:,:]] * self.n, dim=1)], dim=3)
        # x: (bsz*4,2,n_nodes,2* hidden_dim)
        outputs = torch.zeros(x.size()[0],x.size(2),inputs.size(-1)).to(inputs.device)   #(bsz*4,n_nodes,in_features)
        weights_ = torch.zeros(x.size(0),x.size(2),x.size(1), 1).to(inputs.device)  #(bsz*4,n_nodes,2,1)
        for bs in range(x.size()[0]):
            x_ = x[bs]
            U = torch.matmul(x_,self.a).transpose(0, 1)  #(n_nodes,2,1)
            # U = self.norm(U)
            # 非线性激活
            U = F.leaky_relu_(U)
            weights = F.softmax(U, dim=1)
            weights = torch.where(torch.isnan(weights), torch.full_like(weights, 0), weights)
            outputs_bs = torch.matmul(weights.transpose(1, 2), inputs[bs].transpose(0,1)).squeeze(1)*2
            #(n,1,2)*(n,2,in_dim)---(n,in_dim)
            outputs[bs] = outputs_bs
            weights_[bs] = weights
        return outputs, weights_



# class SelfAttention(Module):
#     """docstring for SelfAttention"""
#     def __init__(self, in_features, idx):
#         super(SelfAttention, self).__init__()
#         self.idx = idx
#         self.w = Parameter(torch.FloatTensor(in_features, in_features))
#         self.reset_parameters()

#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.w.size(1))
#         self.w.data.uniform_(-stdv, stdv)

#     def forward(self, inputs):
#         # inputs' shape is like 4*d, w's shape is like d*d
#         # u's shape is like 4*4
#         U = torch.matmul(inputs, self.w)
#         U = torch.matmul(U, inputs.transpose(1, 2))
#         # 非线性激活
#         U = F.tanh(U)
#         weights = F.softmax(U.transpose(0, 1)[self.idx], dim=1)
#         outputs = torch.matmul(torch.stack([weights], dim=1), inputs).squeeze(1)
#         return outputs, weights

class MLP(Module):
    """docstring for MLP"""
    def __init__(self, in_d, out_d):
        super(MLP, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        hidden = in_d / out_d
        hidden1 = int(in_d / math.sqrt(hidden))
        hidden2 = int(hidden1 / math.sqrt(hidden))
        self.l1 = torch.nn.Linear(in_d, hidden1)
        self.l2 = torch.nn.Linear(hidden1, hidden2)
        self.l3 = torch.nn.Linear(hidden2, out_d)

    def forward(self, inputs):

        out = F.relu(self.l1(inputs))
        out = F.relu(self.l2(out))
        out = self.l3(out)
        return out



# class Attention_SupLevel(Module):
#     def __init__(self, W_shape_list, hidden_dim):
#         super(Attention_SupLevel, self).__init__()
#         self.embed = nn.ModuleList()
#         self.W_shape_list = W_shape_list
#         for i in W_shape_list:
#             self.embed.append( torch.nn.Linear( i[0] * i[1], hidden_dim ) )
#         self.w = Parameter(torch.FloatTensor(2 * hidden_dim, 1))
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.w.size(1))
#         self.w.data.uniform_(-stdv, stdv)
#
#     def forward(self, W_list):
#         h = torch.stack([self.embed[i](W_list[i].view(-1)) for i in range(len(W_list))], dim=0)
#         N, M = h.size()
#         a_in = torch.cat( [h.repeat(1, N).view(N*N, -1), h.repeat(N, 1)], dim=1 ).view(N, -1, 2*M)
#         e = F.leaky_relu_(torch.matmul(a_in, a).squeeze(2))
#         weights = F.softmax(e, dim=1)
#         return weights


class GraphAttentionConvolution(Module):
    '''
        output： list（len=type_num） of list(len=type_num) of tensor： size=(bsz*4,n_nodes, out_feature_dim)
        [[11,12],[21,22]]  11,12:n1_nodes, 21,22:n2_nodes
    '''

    def __init__(self, in_features_list, out_features, bias=True, gamma = 0.1):
        super(GraphAttentionConvolution, self).__init__()
        self.ntype = len(in_features_list)
        self.in_features_list = in_features_list
        self.out_features = out_features
        self.weights = nn.ParameterList()
        for i in range(self.ntype):
            cache = Parameter(torch.FloatTensor(in_features_list[i], out_features,))
            nn.init.xavier_normal_(cache.data, gain=1.414)
            self.weights.append(cache)
        # self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
            stdv = 1. / math.sqrt(out_features)
            self.bias.data.uniform_(-stdv, stdv)
            # nn.init.xavier_normal_(self.bias.data, gain=1.414)
        else:
            self.register_parameter('bias', None)
        
        # self.att = Attention_InfLevel(out_features)
        self.att_list = nn.ModuleList()
        for i in range(self.ntype):
            self.att_list.append(Attention_InfLevel(out_features, gamma))


    def forward(self, inputs_list, adj_list, global_W = None):

        h = []
        for i in range(self.ntype):
            h.append(torch.matmul(inputs_list[i], self.weights[i]))  #(bsz*4, n_nodes, embsz) * (embsz, out_feature_dim)

        if global_W is not None:
            for i in range(self.ntype):
                h[i] = (torch.matmul(h[i], global_W) )
        outputs = []
        for t1 in range(self.ntype):
            x_t1 = []
            for t2 in range(self.ntype):

                # if adj_list[t1][t2] == :
                #     x_t1.append(torch.zeros(adj_list[t1][t2].shape[0], self.out_features, device=self.bias.device))
                #     continue
                #     # print('error.')
                #
                if self.bias is not None:
                    # x_t1.append( self.att(h[t1], h[t2], adj_list[t1][t2]) + self.bias )
                    x_t1.append(self.att_list[t1](h[t1], h[t2], adj_list[t1][t2]) + self.bias)
                else:
                    # x_t1.append( self.att(h[t1], h[t2], adj_list[t1][t2]) )
                    x_t1.append(self.att_list[t1](h[t1], h[t2], adj_list[t1][t2]))
            outputs.append(x_t1)
            
        return outputs
class Attention_InfLevel(nn.Module):
    '''
    计算input1 和 input2 之间的attention，并返回input1的加权更新值(维度为dim_features)
    input1 & input2 :(bsz*4, n_nodes, out_feature_dim)
    adj_matrix: (bsz*4,n1_nodes, n2_nodes)
    Here, dim_feature = out_feature_dim
    return:
        (bsz*4,n1_nodes,out_feature_dim)
    '''
    def __init__(self, dim_features, dropout = 0.1):
        super(Attention_InfLevel, self).__init__()
        self.dim_features = dim_features
        # self.alpha = alpha
        # self.concat = concat
        
        self.a1 = nn.Parameter(torch.zeros(size=(dim_features, 1)))
        self.a2 = nn.Parameter(torch.zeros(size=(dim_features, 1)))
        nn.init.xavier_normal_(self.a1.data, gain=1.414)
        nn.init.xavier_normal_(self.a2.data, gain=1.414)        

        # self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(0.2)

        # att = mul(att, adj) * γ + adj * (1 - γ)
        # γ = 0 : no attention;    γ = 1 : no original adj
        self.dropout = dropout

    
    def forward(self, input1, input2, adj):
        h = input1  #size = (bsz*4,n1_nodes,out_dim)
        g = input2  #size = (bsz*4,n2_nodes,out_dim)
        N = h.size()[1]
        M = g.size()[1]

        e1 = torch.matmul(h, self.a1).repeat(1, 1, M)  #size = (bsz*4,N,M)
        e2 = torch.matmul(g, self.a2).transpose(1,2).repeat(1, N, 1)  #size = (bsz*4,N,M)
        e = e1 + e2
        e = self.leakyrelu(e)  #size = (bsz*4,n1_nodes,n2_nodes)
        
        zero_vec = -9e15 * torch.ones_like(e).to(input1.device)
        # zero_vec = torch.zeros_like(e)
        attention = torch.where(adj > 0, e, zero_vec).to(input1.device)
        attention = F.softmax(attention, dim=2)
        attention = torch.where(torch.isnan(attention), torch.full_like(attention, 0), attention)
        adj_ = torch.zeros_like(attention).to(input1.device)
        for bs in range(adj_.size(0)):
            adj_[bs] = adj[bs].sum(1).repeat(M, 1).T  #[n1,n2]
        attention = torch.mul(attention, adj_)  # (bsz*4,n1_nodes, n2_nodes)对应位相乘
        # attention = F.dropout(attention, self.dropout, training=self.training)

        del(zero_vec)

        # attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, g)  # (bsz*4,n1_nodes, n2_nodes) * (bsz*4, n2_nodes, out_feature_dim)
        return h_prime  #(bsz*4,n1_nodes,out_feature_dim)

class FFNLayer(nn.Module):  #全连接层，由input_dim → intermediate_dim → output_dim
    def __init__(self, input_dim, intermediate_dim, output_dim, dropout, layer_norm=True):
        super(FFNLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        if layer_norm:
            self.ln = nn.LayerNorm(intermediate_dim)
        else:
            self.ln = None
        self.dropout_func = nn.Dropout(dropout)
        self.fc2 = nn.Linear(intermediate_dim, output_dim)
        self.activate = nn.Tanh()

    def forward(self, input):
        inter = self.fc1(self.dropout_func(input))
        inter_act = self.activate(inter)
        if self.ln:
            inter_act = self.ln(inter_act)
        return self.fc2(inter_act)



class ResidualGRU(nn.Module):
    def __init__(self, hidden_size, dropout=0.1, num_layers=2):
        super(ResidualGRU, self).__init__()
        self.enc_layer = nn.GRU(input_size=hidden_size, hidden_size=hidden_size // 2, num_layers=num_layers,
                                batch_first=True, dropout=dropout, bidirectional=True)
        self.enc_ln = nn.LayerNorm(hidden_size)

    def forward(self, input):
        output, _ = self.enc_layer(input)
        return self.enc_ln(output + input)


class ArgumentGCN(nn.Module):

    def __init__(self, node_dim, extra_factor_dim=0, iteration_steps=1):
        super(ArgumentGCN, self).__init__()
        self.node_dim = node_dim
        self.iteration_steps = iteration_steps
        self._node_weight_fc = torch.nn.Linear(node_dim + extra_factor_dim, 1, bias=True)
        self._self_node_fc = torch.nn.Linear(node_dim, node_dim, bias=True)
        self._node_fc_argument = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._node_fc_punctuation = torch.nn.Linear(node_dim, node_dim, bias=False)

    def forward(self,
                node,
                node_mask,
                argument_graph,
                punctuation_graph,
                extra_factor=None):
        ''' '''
        '''
        Current: 2 relation patterns.
            - argument edge. (most of them are causal relations)
            - punctuation edges. (including periods and commas)
        node size = (bsz x n_choices, n_nodes, embed_size) 
        node_masks size = (bsz x n_choices, max_n_nodes)
        '''

        node_len = node.size(1) # n_nodes

        diagmat = torch.diagflat(torch.ones(node.size(1), dtype=torch.long, device=node.device))  # 生成对角线为1的方阵 n * n, n = n_nodes
        diagmat = diagmat.unsqueeze(0).expand(node.size(0), -1, -1)  # size = (bsz x n_choices, n_nodes, n_nodes) 第一维方向上复制
        dd_graph = node_mask.unsqueeze(-1) * node_mask.unsqueeze(1) * (1 - diagmat)
        #(bsz,1,n_nodes)*(bsz,n_nodes,1) = (bsz,n_node,n_node)  对角线处取0，非对角线处
        graph_argument = dd_graph * argument_graph
        graph_punctuation = dd_graph * punctuation_graph
        # graph_argument, graph_punctuation shape = (bsz x n_choices, n_node, n_node)
        node_neighbor_num = graph_argument.sum(-1) + graph_punctuation.sum(-1)  # shape=(bsz x n_choices, n_nodes)
        node_neighbor_num_mask = (node_neighbor_num >= 1).long()  #有neighbor连接的点为1，没连接的为0  shape=(bsz x n_choices, n_nodes)
        node_neighbor_num = util.replace_masked_values(node_neighbor_num.float(), node_neighbor_num_mask, 1)  # 没有边连接的点都记为1个边连接

        all_weight = []
        for step in range(self.iteration_steps):

            ''' (1) Node Relatedness Measure '''
            if extra_factor is None:
                d_node_weight = torch.sigmoid(self._node_weight_fc(node)).squeeze(-1)  #(bsz x n_choices, n_nodes) 映到1维，并删去最后一维的表示
            else:
                d_node_weight = torch.sigmoid(self._node_weight_fc(torch.cat((node, extra_factor), dim=-1))).squeeze(-1)

            all_weight.append(d_node_weight)   #每个step都append一次weight

            self_node_info = self._self_node_fc(node)  # (bsz x n_choices, n_nodes, embed_size)

            ''' (2) Message Propagation (each relation type) '''
            # 获取 argument_graph的更新信息
            node_info_argument = self._node_fc_argument(node) # (bsz x n_choices, n_nodes, emb_size)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1), #(bsz x n_choices, n_nodes, n_nodes) 第二维复制
                graph_argument, 0)     #前tensor 在 graph_argument 中被masked的位置处用0替代
            node_info_argument = torch.matmul(node_weight, node_info_argument) # (bsz x n_choices, n_nodes, emb_size)

            # 获取 punctuation_graph的更新信息
            node_info_punctuation = self._node_fc_punctuation(node)
            node_weight = util.replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graph_punctuation, 0)
            node_info_punctuation = torch.matmul(node_weight, node_info_punctuation) # (bsz x n_choices, n_nodes, emb_size)


            agg_node_info = (node_info_argument + node_info_punctuation) / node_neighbor_num.unsqueeze(-1)
            # shape=(bsz x n_choices, n_nodes, embed_size) / (bsz x n_choices, n_nodes, 1)
            ''' (3) Node Representation Update '''
            node = F.relu(self_node_info + agg_node_info) # node shape = (bsz x n_choices, n_nodes, embed_size)

        all_weight = [weight.unsqueeze(1) for weight in all_weight]  # shape of items = (bsz x n_choices, 1, n_nodes)
        all_weight = torch.cat(all_weight, dim=1) # shape= (bsz x n_choices, iteration_steps, n_nodes)

        return node, all_weight






def masked_softmax(vector: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = -1,
                   memory_efficient: bool = True,#False,
                   mask_fill_value: float = -1e32) -> torch.Tensor:
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).bool(), mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result


def weighted_sum(matrix: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
    # We'll special-case a few settings here, where there are efficient (but poorly-named)
    # operations in pytorch that already do the computation we need.
    if attention.dim() == 2 and matrix.dim() == 3:
        return attention.unsqueeze(1).bmm(matrix).squeeze(1)
    if attention.dim() == 3 and matrix.dim() == 3:
        return attention.bmm(matrix)
    if matrix.dim() - 1 < attention.dim():
        expanded_size = list(matrix.size())
        for i in range(attention.dim() - matrix.dim() + 1):
            matrix = matrix.unsqueeze(1)
            expanded_size.insert(i + 1, attention.size(i + 1))
        matrix = matrix.expand(*expanded_size)
    intermediate = attention.unsqueeze(-1).expand_as(matrix) * matrix
    return intermediate.sum(dim=-2)


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        # 创建空向量并初始化  W矩阵：infea → outfea
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        # 创建空向量并初始化  a矩阵： （2*outfea） * 1
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):  # adj：邻接矩阵
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        # 实现Whi和Whj concatenation  shape  = (N, N, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # 激活后去掉最后两个维度

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # 不邻接的元素设为-∞
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)  # shape = (N, out_features)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # number of nodes

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks):
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        #
        # These are the rows of the second matrix (Wh_repeated_alternating):
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN
        # '----------------------------------------------------' -> N times
        #

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'