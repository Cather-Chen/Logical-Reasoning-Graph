import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import numpy as np
from typing import List, Dict, Any, Tuple
from itertools import groupby
from operator import itemgetter
import copy
from transformers import BertPreTrainedModel, RobertaModel,ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP,RobertaConfig
from models import HGAT,GAT
from layers import FFNLayer,masked_softmax,weighted_sum,ResidualGRU,ArgumentGCN
from eval_attention import SCAttention


class MyHGAT(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"
    def __init__(self,
                 config,
                 max_rel_id,
                 feature_dim_list,
                 device,
                 use_pool: bool = True,
                 dropout_prob: float = 0.1,
                 token_encoder_type: str = "roberta",
                 ) -> None:
        super(MyHGAT, self).__init__(config)

        self.token_encoder_type = token_encoder_type
        self.max_rel_id = max_rel_id
        self.roberta = RobertaModel(config)
        self.use_pool = use_pool
        if self.use_pool:
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.device = device
        self.HGAT = HGAT(nfeat_list=feature_dim_list, nhid=config.hidden_size, out_dim=config.hidden_size, dropout=dropout_prob)
        self.GAT = GAT(nfeat=config.hidden_size,nhid=config.hidden_size,nclass=config.hidden_size)
        self.MLP = FFNLayer(config.hidden_size*3, config.hidden_size, 1, dropout_prob)
        self.attention = SCAttention(config.hidden_size,config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size)
        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self._proj_sequence_h = nn.Linear(config.hidden_size, 1, bias=False)
        self.classifer = nn.Linear(config.hidden_size, 1)
        self.init_weights()
        self.gru = ResidualGRU(config.hidden_size, dropout_prob, 2)
        self.gcn = ArgumentGCN(node_dim=config.hidden_size, iteration_steps=2)
    def split_into_spans_9(self, seq, seq_mask, split_bpe_ids):
        '''

            :param seq: (bsz, seq_length, embed_size)
            :param seq_mask: (bsz, seq_length)
            :param split_bpe_ids: (bsz, seq_length). value = {-1, 0, 1, 2, 3, 4}.
            :return:
                - encoded_spans: (bsz, n_nodes, embed_size)
                - span_masks: (bsz, n_nodes)
                - edges: (bsz, n_nodes - 1)
                - node_in_seq_indices: list of list of list(len of span).

        '''

        def _consecutive(seq: list, vals: np.array):
            groups_seq = []
            output_vals = copy.deepcopy(vals)
            for k, g in groupby(enumerate(seq), lambda x: x[0] - x[1]):
                groups_seq.append(list(map(itemgetter(1), g)))
            output_seq = []
            for i, ids in enumerate(groups_seq):
                output_seq.append(ids[0])
                if len(ids) > 1:
                    output_vals[ids[0]:ids[-1] + 1] = min(output_vals[ids[0]:ids[-1] + 1])
            return groups_seq, output_seq, output_vals

        embed_size = seq.size(-1)
        device = seq.device
        encoded_spans = []
        span_masks = []
        edges = []
        node_in_seq_indices = []
        for item_seq_mask, item_seq, item_split_ids in zip(seq_mask, seq, split_bpe_ids):
            item_seq_len = item_seq_mask.sum().item()
            item_seq = item_seq[:item_seq_len]
            item_split_ids = item_split_ids[:item_seq_len]
            item_split_ids = item_split_ids.detach().cpu().numpy()
            split_ids_indices = np.where(item_split_ids > 0)[0].tolist()
            grouped_split_ids_indices, split_ids_indices, item_split_ids = _consecutive(
                split_ids_indices, item_split_ids)
            n_split_ids = len(split_ids_indices)

            item_spans, item_mask = [], []
            item_edges = []
            item_node_in_seq_indices = []
            item_edges.append(item_split_ids[split_ids_indices[0]])
            for i in range(n_split_ids):
                if i == n_split_ids - 1:
                    span = item_seq[split_ids_indices[i] + 1:]
                    if not len(span) == 0:
                        item_spans.append(span.sum(0))
                        item_mask.append(1)

                else:
                    span = item_seq[split_ids_indices[i] + 1:split_ids_indices[i + 1]]
                    if not len(span) == 0:
                        item_spans.append(span.sum(0))
                        item_mask.append(1)
                        item_edges.append(item_split_ids[split_ids_indices[i + 1]])
                        item_node_in_seq_indices.append([i for i in range(grouped_split_ids_indices[i][-1] + 1,
                                                                          grouped_split_ids_indices[i + 1][0])])

            encoded_spans.append(item_spans)
            span_masks.append(item_mask)
            edges.append(item_edges)
            node_in_seq_indices.append(item_node_in_seq_indices)

        max_nodes = max(map(len, span_masks))
        span_masks = [spans + [0] * (max_nodes - len(spans)) for spans in span_masks]
        span_masks = torch.from_numpy(np.array(span_masks))
        span_masks = span_masks.to(device).long()

        pad_embed = torch.zeros(embed_size, dtype=seq.dtype, device=seq.device)
        encoded_spans = [spans + [pad_embed] * (max_nodes - len(spans)) for spans in encoded_spans]
        encoded_spans = [torch.stack(lst, dim=0) for lst in encoded_spans]
        encoded_spans = torch.stack(encoded_spans, dim=0)
        encoded_spans = encoded_spans.to(device).float()
        truncated_edges = [item[1:-1] for item in edges]

        return encoded_spans, span_masks, truncated_edges, node_in_seq_indices

    def get_adjacency_matrices_2(self, edges:List[List[int]], n_nodes:int, device):
        '''
        Convert the edge_value_list into adjacency matrices.
            * argument graph adjacency matrix. Asymmetric (directed graph).
            * punctuation graph adjacency matrix. Symmetric (undirected graph).

            : argument
                - edges:list[list[str]]. len_out=(bsz x n_choices), len_in=n_edges. value={-1, 0, 1, 2, 3, 4, 5}.

            Note: relation patterns
                1 - (relation, head, tail) 
                2 - (head, relation, tail) 
                3 - (tail, relation, head)
                4 - (head, relation, tail) & (tail, relation, head) 
                5 - (head, relation, tail) & (tail, relation, head) 

        '''

        batch_size = len(edges)
        argument_graph = torch.zeros(
            (batch_size, n_nodes, n_nodes)) 
        punct_graph = torch.zeros(
            (batch_size, n_nodes, n_nodes))  
        for b, sample_edges in enumerate(edges):
            for i, edge_value in enumerate(sample_edges):
                if edge_value == 1:  
                    try:
                        argument_graph[b, i + 1, i + 2] = 1
                    except Exception:
                        pass
                elif edge_value == 2:  
                    argument_graph[b, i, i + 1] = 1
                elif edge_value == 3:  
                    argument_graph[b, i + 1, i] = 1
                elif edge_value == 4: 
                    argument_graph[b, i, i + 1] = 1
                    argument_graph[b, i + 1, i] = 1
                elif edge_value == 5:  
                    try:
                        punct_graph[b, i, i + 1] = 1
                        punct_graph[b, i + 1, i] = 1
                    except Exception:
                        pass
        return argument_graph.to(device), punct_graph.to(device)
        # return a sparse adj matrix

    def creat_SVO_id_batch(self,SVO_ids):
        '''
        input:
            SVO_ids: size :tensor, size = (bsz*4,padding=16)
        return:
            SVO_id_batch: list(len=bsz) of list(len=padding=16) of list(1):[[[xx],[xx]...x16],...]
        '''
        SVO_id_batch = []
        bsz = SVO_ids.size(0)
        for bs in range(bsz):
            SVO_id_list = []
            for id in SVO_ids[bs, :].detach().cpu().numpy():
                if id != 1:
                    SVO_id_list.append([id])
                else:
                    SVO_id_list.append([])
            SVO_id_batch.append(SVO_id_list)
        return SVO_id_batch

    def create_keywords_feature(self, emb, segid, flat_keyword_ids,device):
        '''
        input:
            keywords embedding:(4*bsz, padding size=16, embed_size)
            segids:(bsz*4,padding=16)
        return:
            embed_batch: tensor size = (bsz, max_nodes,embsz)
            keytoken_batch: list(len=bsz) of list(len=max_nodes) of list(len = keytoken_num)
        '''
        embed_batch = []
        keytoken_batch = []
        length = segid.shape[0]
        for t in range(length):
            keytoken_list = []
            embed = []
            seglist = segid[t, :].detach().cpu().numpy()
            seglist = np.concatenate(([0],seglist))
            for i in range(len(seglist) - 1):
                start = seglist[i]
                end = seglist[i + 1]
                if end == -1:
                    break
                if start == end:
                    continue
                embed_tensor = torch.mean(emb[t, start:end, :], dim=0)   # [fea_dim]
                keytoken_list.append(flat_keyword_ids[t, start:end].detach().cpu().numpy())
                embed.append(embed_tensor)
            keytoken_batch.append(keytoken_list)  
            embed_batch.append(embed)
        max_len = max(map(len, embed_batch))
        a = torch.zeros(emb.size(-1), dtype=emb.dtype, device=device)
        b = []
        embed_batch = [spans + [a] * (max_len - len(spans)) for spans in embed_batch]  # list (bsz) of list(max) of tensor(feadim)
        keyword_feature = torch.zeros((len(embed_batch),max_len,emb.size(-1))).to(device)
        for j in range(len(embed_batch)):
            for i in range(max_len):
                keyword_feature[j,i,:] = embed_batch[j][i]

        keytoken_batch = [spans + [b] * (max_len - len(spans)) for spans in keytoken_batch]
        keyword_feature = torch.where(torch.isnan(keyword_feature), torch.full_like(keyword_feature, 0), keyword_feature)
        return keyword_feature, keytoken_batch

    def create_graph(self, sequence_output, flat_punct_bpe_ids, flat_argument_bpe_ids, flat_attention_mask, flat_input_ids,device):
        '''
        return:
            -encoded_spans : (bsz, n_nodes, embed_size)
            -graph : (bsz n_nodes, n_nodes)
            -ids_in_sentence_nodes : list(len=bsz) of list(len = max_nodes) of list(len=ids_num)
        '''
        new_punct_id = self.max_rel_id + 1
        new_punct_bpe_ids = new_punct_id * flat_punct_bpe_ids  # punct_id: 1 -> 5. for incorporating with argument_bpe_ids.
        _flat_all_bpe_ids = flat_argument_bpe_ids + new_punct_bpe_ids
        overlapped_punct_argument_mask = (_flat_all_bpe_ids > new_punct_id).long()
        flat_all_bpe_ids = _flat_all_bpe_ids * (
                    1 - overlapped_punct_argument_mask) + flat_argument_bpe_ids * overlapped_punct_argument_mask
        assert flat_argument_bpe_ids.max().item() <= new_punct_id

        encoded_spans, span_mask, edges, node_in_seq_indices = self.split_into_spans_9(sequence_output,
                                                                                       flat_attention_mask,
                                                                                       flat_all_bpe_ids)
        # encoded_spans: (bsz x n_choices, n_nodes, embed_size)  here，embedding vectors have been taken the average
        # span_masks: (bsz x n_choices, n_nodes)
        # edges:list[list[str]]. len_out=(bsz x n_choices), len_in=n_edges=n_nodes - 1. value={-1, 0, 1, 2, 3, 4, 5}
        # node_in_seq_indices: list(len=bsz) of list(len=n_nodes) of list(len=varied).
        bsz = encoded_spans.size(0)
        max_node = encoded_spans.size(1)
        ids_in_sentence_nodes = []
        for bs in range(bsz):
            ids_in_sentence_nodes_list = []
            for t in range(max_node):
                ids_in_sentence_nodes_list.append([])
            bpe_list = flat_all_bpe_ids[bs, :].detach().cpu().numpy()
            input_ids_list = flat_input_ids[bs, :].detach().cpu().numpy()  #size = [bsz*4,seq_length]
            k = 0
            for i in range(len(bpe_list)):   # -1:padding, 0:non, 4: arg, 5:punct.
                if bpe_list[i] == 0:
                    if k <= max_node-1:
                        ids_in_sentence_nodes_list[k].append(input_ids_list[i])
                    continue
                if bpe_list[i] == -1:
                    break
                if bpe_list[i] == 4:
                    if k <= max_node -1 and ids_in_sentence_nodes_list[k] == []:
                        ids_in_sentence_nodes_list[k].append(input_ids_list[i])
                        continue
                    else:
                        if k<=max_node-2:
                            k += 1
                            ids_in_sentence_nodes_list[k].append(input_ids_list[i])
                        continue
                if bpe_list[i] == 5:
                    if k <=max_node-1 and ids_in_sentence_nodes_list[k] == []:
                        continue
                    else:
                        k += 1
                        continue
            ids_in_sentence_nodes.append(
                ids_in_sentence_nodes_list)  # list(len=bsz) of list(len = max_nodes) of list(len=ids_num)

        argument_graph, punctuation_graph = self.get_adjacency_matrices_2(edges, n_nodes=encoded_spans.size(1),
                                                                          device=encoded_spans.device)
        #torch.size (batch_size, n_nodes, n_nodes)
        eyes = torch.eye(max_node,max_node).unsqueeze(0).repeat(bsz,1,1).to(device)
        graph = argument_graph + punctuation_graph + eyes #torch.size (batch_size, n_nodes, n_nodes)
        graph_mask = (graph > 0)
        graph = torch.ones_like(argument_graph) * graph_mask + torch.zeros_like(argument_graph) * (~graph_mask)
        graph = graph.to(device)
        # (batch_size, n_nodes, n_nodes)

        return encoded_spans, graph, ids_in_sentence_nodes,span_mask,node_in_seq_indices,argument_graph,punctuation_graph

    def create_similar_adj(self,feature_word,device):
        '''
        input:
            feature_word: size = (4*bsz, padding+max_nodes,embsz)  padding:SVO,max_nodes:key_words
            adj_SVO : size = [bsz,4,padding,max]
        return:
            similar_adj: size = (4*bsz, padding+max_nodes,padding+max_nodes)
        '''
        bsz = feature_word.size(0)
        length = feature_word.size(1)
        adj_matrix = torch.zeros((bsz, length, length)).to(device)
        # flat_adj_SVO = adj_SVO.view(-1,adj_SVO.size(2), adj_SVO.size(-1))  # [bsz*4,padding,max]

        # padding_size = adj_SVO.size(-2)

        def get_cos_similar(v1, v2):
            num = torch.dot(v1,v2).item()
            denom = torch.linalg.norm(v1).item() * torch.linalg.norm(v2).item()
            return num / denom if denom != 0 else 0

        for bs in range(bsz):
            feature_batch =feature_word[bs]
            # adj_SVO_batch = flat_adj_SVO[bs]   #tensor,size =(padding,max)
            adj_matrix_batch = torch.zeros((length,length)).to(device)
            for i in range(length-1):
                word_i = feature_batch[i]
                if (word_i == 0).all():
                    continue
                for j in range(i+1, length):
                    word_j = feature_batch[j]
                    if (word_j == 0).all():
                        continue
                    cos_similarity = get_cos_similar(word_i,word_j)
                    if cos_similarity > 0.5:
                        adj_matrix_batch[i][j] = 1


            adj_matrix_batch = adj_matrix_batch + adj_matrix_batch.T + torch.eye(length,length).to(device)

            # for i in range(padding_size):  #tensor,size =(padding,max)
            #     adj_list = adj_SVO_batch[i].detach().cpu().numpy()
            #     for j in adj_list:
            #         if j == -1:
            #             break
            #         if j != -1:
            #             adj_matrix_batch[i][j] = 1


            adj_matrix[bs] = adj_matrix_batch
        return adj_matrix

    def create_word2sentence(self,word_ids_batch,ids_in_sentence_nodes,device):
        '''
            input:
                -word_ids_batch: list(len=bsz*4) of list(len=16+max_nodes1) of list
                -ids_in_sentence_nodes: list(len=bsz*4) of list(len = max_nodes2) of list(len=ids_num)
            return:
                -word2sent_adj: tensor size = (bsz*4, 16+max_nodes1,max_nodes2)
                -sent2word_adj: tensor size = (bsz*4, max_nodes2, 16+max_nodes1)
        '''
        assert len(word_ids_batch) == len(ids_in_sentence_nodes)
        bsz = len(word_ids_batch)
        word_nodes_num = len(word_ids_batch[0])
        sent_nodes_num = len(ids_in_sentence_nodes[0])
        word2sent_adj = torch.zeros(bsz,word_nodes_num,sent_nodes_num).to(device)
        sent2word_adj = torch.zeros(bsz,sent_nodes_num, word_nodes_num).to(device)
        def inter(a, b):
            return list(set(a) & set(b))
        for bs in range(bsz):
            adj_batch = torch.zeros(word_nodes_num, sent_nodes_num).to(device)
            word_ids = word_ids_batch[bs]
            sent_ids = ids_in_sentence_nodes[bs]
            for i in range(word_nodes_num):
                for j in range(sent_nodes_num):
                    if inter(word_ids[i],sent_ids[j]) != []:
                        adj_batch[i][j] = 1

            word2sent_adj[bs] = adj_batch
            sent2word_adj[bs] = adj_batch.T

        return word2sent_adj.to(device), sent2word_adj.to(device)

    def create_sent2sent_type3(self,word2sent_adj,device):
        '''
        input:
            -word2sent_adj: tensor,(bsz*4,word_node_num,sent_node_num)
        return:
            -sent2sent_adj: tensor(bsz*4,sent_node_num,sent_node_num)
        '''
        bsz = word2sent_adj.size(0)
        sent_node_num = word2sent_adj.size(-1)
        word_node_num = word2sent_adj.size(-2)
        sent2sent_adj = torch.zeros(bsz,sent_node_num,sent_node_num).to(device)
        for bs in range(bsz):
            sent2sent_adj_batch = torch.zeros(sent_node_num,sent_node_num).to(device)
            word2sent_adj_batch = word2sent_adj[bs]
            for i in range(word_node_num):
                word2sent = word2sent_adj_batch[i].detach().cpu().numpy()
                adj_list = []
                for j in range(len(word2sent)):
                    if word2sent[j] == 1:
                        adj_list.append(j)
                length = len(adj_list)
                for t in range(length-1):
                    for s in range(t+1,length):
                        sent2sent_adj_batch[adj_list[t]][adj_list[s]] = 1
                        sent2sent_adj_batch[adj_list[s]][adj_list[t]] = 1
            sent2sent_adj_batch = sent2sent_adj_batch + torch.eye(sent_node_num,sent_node_num).to(device)
            sent2sent_adj[bs] = sent2sent_adj_batch
        return sent2sent_adj.to(device)

    def get_gcn_info_vector(self, indices, node, size, device):
        '''

        :param indices: list(len=bsz) of list(len=n_nodes) of list(len=varied).
        :param node: (bsz, n_nodes, embed_size)
        :param size: value=(bsz, seq_len, embed_size)
        :param device:
        :return:
            - gcn_info_vec: shape = (bsz, seq_len, embed_size)
        '''

        batch_size = size[0]
        gcn_info_vec = torch.zeros(size=size, dtype=torch.float, device=device)

        for b in range(batch_size):
            for ids, emb in zip(indices[b], node[b]): #there are n_nodes in b-th indices, ids and emb mean the id and embedding of a certain node.
                gcn_info_vec[b, ids] = emb

        return gcn_info_vec




    def forward(self,
                input_ids: torch.LongTensor,
                attention_mask: torch.LongTensor,
                argument_bpe_ids: torch.LongTensor,
                punct_bpe_ids: torch.LongTensor,
                keytokensids: torch.LongTensor,
                keymask: torch.LongTensor,
                key_segid: torch.LongTensor,
                SVO_ids: torch.LongTensor,  #[bsz,4,16]
                SVO_mask: torch.LongTensor,  #[bsz,4,16]
                adj_SVO: torch.LongTensor,   #[bsz,4,16,max]
                labels: torch.LongTensor,
                passage_mask:torch.LongTensor,
                question_mask: torch.LongTensor,
                # token_type:torch.LongTensor
                ) -> Tuple:

        num_choices = input_ids.shape[1]  # input_ids.shape = (bsz,4,seq_length)
        # [bsz,4,?]-->(bsz x 4, ?)
        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_key_segid = key_segid.view(-1, key_segid.size(-1)) if key_segid is not None else None
        # flat_token_type_ids = token_type.view(-1, token_type.size(-1)) if token_type is not None else None
        flat_passage_mask = passage_mask.view(-1, passage_mask.size(-1)) if passage_mask is not None else None
        flat_question_mask = question_mask.view(-1, question_mask.size(-1)) if question_mask is not None else None
        # flat_domain_bpe_ids = domain_bpe_ids.view(-1, domain_bpe_ids.size(-1)) if domain_bpe_ids is not None else None
        flat_argument_bpe_ids = argument_bpe_ids.view(-1, argument_bpe_ids.size(-1)) if argument_bpe_ids is not None else None
        flat_punct_bpe_ids = punct_bpe_ids.view(-1, punct_bpe_ids.size(-1)) if punct_bpe_ids is not None else None
        flat_SVO_ids = SVO_ids.view(-1,SVO_ids.size(-1)) if SVO_ids is not None else None
        flat_keyword_ids = keytokensids.view(-1,keytokensids.size(-1)) if keytokensids is not None else None
        flat_SVO_mask = SVO_mask.view(-1,SVO_mask.size(-1)) if SVO_mask is not None else None
        flat_keyword_mask = keymask.view(-1,keymask.size(-1)) if keymask is not None else None

        ## baseline embedding
        corpus_outputs = self.roberta(flat_input_ids, attention_mask=flat_attention_mask)
        sequence_output = corpus_outputs[0]  # shape =  (4*bsz, seq_length, embed_size)
        pooled_output = corpus_outputs[1]  # shape = (4*bsz, emb_size)


        # SVO embedding + node truncation
        SVO_outputs = self.roberta(flat_SVO_ids, attention_mask=flat_SVO_mask, token_type_ids=None)
        SVO_emb = SVO_outputs[0]  # (4*bsz, padding size=16, embed_size)
        SVO_id_batch = self.creat_SVO_id_batch(flat_SVO_ids)
        SVO_node_mask = torch.zeros((len(SVO_id_batch), len(SVO_id_batch[0]))).to(self.device)  # [bsz,n_nodes]
        for bs in range(SVO_node_mask.size(0)):
            for j in range(len(SVO_id_batch[bs])):  # list(len=bsz*4) of list(len=16+max_nodes) of list
                if SVO_id_batch[bs][j] != []:
                    SVO_node_mask[bs, j] = 1
        SVO_truncation = int(max(SVO_node_mask.sum(1)).item())
        SVO_node_mask = SVO_node_mask[:, :SVO_truncation]
        SVO_emb = SVO_emb[:, :SVO_truncation, :]
        SVO_id_batch = [span[:SVO_truncation] for span in SVO_id_batch]


        ## EDU embedding + sent2sent
        encoded_spans, graph, ids_in_sentence_nodes, sent_mask,node_in_seq_indices,argument_graph, punct_graph = self.create_graph(sequence_output, flat_punct_bpe_ids,
                                                                   flat_argument_bpe_ids, flat_attention_mask,
                                                                   flat_input_ids, device=self.device)
        # sent_mask: (bsz,n_nodes)
        # graph size: (bsz, n_nodes, n_nodes)
        # word_ids_batch = [SVO_id_batch[bs]+key_id_batch[bs] for bs in range(len(SVO_id_batch))]  #list(len=bsz*4) of list(len=16+max_nodes) of list
        # feature_word = torch.cat((SVO_emb,embed_batch), dim=1).to(self.device)  #size = (4*bsz, padding+max_nodes,embsz)
        # word_ids_batch = [key_id_batch[bs] for bs in range(len(key_id_batch))]

        feature_list = [encoded_spans, SVO_emb]   # list(len=2) [sentence_feature, words_feature]

        words2words_adj = self.create_similar_adj(SVO_emb,device=self.device)
        word2sent_adj, sent2word_adj = self.create_word2sentence(SVO_id_batch, ids_in_sentence_nodes,device=self.device)  #tensor size = (bsz*4, 16+max_nodes1,max_nodes2)
        sent2sent_adj3 = self.create_sent2sent_type3(word2sent_adj,device=self.device) #tensor(bsz*4,sent_node_num,sent_node_num)

        assert graph.size() == sent2sent_adj3.size()

        sent2sent_adj = graph + sent2sent_adj3
        sent2sent_mask = (sent2sent_adj > 0)
        sent2sent_adj = torch.zeros_like(sent2sent_adj) * (~sent2sent_mask) + torch.ones_like(sent2sent_adj) * sent2sent_mask
        sent2sent_adj = sent2sent_adj.to(self.device)
        type_sent_adj = [sent2sent_adj, sent2word_adj]
        type_word_adj = [word2sent_adj,words2words_adj]
        adj_list = [type_sent_adj, type_word_adj]
        graph_out = self.HGAT(feature_list, adj_list)# list(len = n_type) of tensor(bsz * 4, n_nodes, out_dim)

        sent_rep = graph_out[0]
        word_rep = graph_out[1]  #(bsz*4,n2_nodes,out_dim)

        # node, node_weight = self.gcn(node=encoded_spans, node_mask=sent_mask,
        #                               argument_graph=argument_graph,
        #                               punctuation_graph=punct_graph)
        graph_info_vec = self.get_gcn_info_vector(node_in_seq_indices, sent_rep, size=sequence_output.size(), device=sequence_output.device)
        edu_update_sequence_output = self.gru(self.norm(sequence_output + graph_info_vec))


        sequence_output = edu_update_sequence_output
        sequence_h2_weight = self._proj_sequence_h(sequence_output).squeeze(-1)  # (bsz, seq_length)
        passage_h2_weight = masked_softmax(sequence_h2_weight.float(), flat_passage_mask.float())
        passage_h2 = weighted_sum(sequence_output, passage_h2_weight) # (bsz,hidden_size)
        question_h2_weight = masked_softmax(sequence_h2_weight.float(), flat_question_mask.float())
        question_h2 = weighted_sum(sequence_output, question_h2_weight)  # (bsz,hidden_size)


        pooled_output = pooled_output.unsqueeze(1)
        delta_att = self.attention(pooled_output, word_rep, SVO_node_mask).to(self.device)
        pooled_output = pooled_output + delta_att
        pooled_output = pooled_output.squeeze(1)

         #[bsz,hidden_size*3]
        if self.use_pool:
            pooled_output = self.dropout(pooled_output) # (bsz*4,hidden_dim)

        output_feats = torch.cat([passage_h2, question_h2, pooled_output], dim=1)
        logit = self.MLP(output_feats)
        reshaped_logits = logit.view(-1, num_choices)  # shape= (bsz,num_choices)
        outputs = (reshaped_logits,)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs
        return outputs






