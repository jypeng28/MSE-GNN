import torch
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import reset
nn_act = torch.nn.ReLU()
F_act = F.relu
from utils import *
from encoder import GNN_encoder


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, device, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.device = device
    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:

        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        features = F.normalize(features)
        device = self.device
        # pdb.set_trace()
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        
        logits = anchor_dot_contrast - logits_max.detach()
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss



class Model(torch.nn.Module):
    def __init__(self, args):
    # def __init__(self, num_tasks, num_layer = 3, node_feature_size = 2, emb_dim = 128, gnn_type = 'gin', drop_ratio = 0.5, gamma = 0.4, use_linear_predictor=False):
        '''
            num_tasks (int): number of labels to be predicted
        '''

        super(Model, self).__init__()
        self.num_classes = args.way
        self.num_layer = args.num_layer
        self.drop_ratio = args.drop_ratio
        self.emb_dim = args.emb_dim
        self.gamma  = args.gamma
        self.node_feature_size = args.num_features
        self.device = args.device
        
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        self.gnn_type = args.gnn
        self.emb_dim_rat = self.emb_dim

        rationale_gnn_node = GNN_encoder(2, self.emb_dim_rat, drop_ratio = self.drop_ratio, gnn_name = self.gnn_type, node_feature_size = self.node_feature_size)
        self.graph_encoder = GNN_encoder(self.num_layer, self.emb_dim, drop_ratio = self.drop_ratio, gnn_name = self.gnn_type, node_feature_size = self.node_feature_size)

        self.separator = separator(
            rationale_gnn_node, 
            torch.nn.Sequential(torch.nn.Linear(self.emb_dim_rat*3, 2*self.emb_dim_rat), torch.nn.BatchNorm1d(2*self.emb_dim_rat,track_running_stats=False), nn_act, torch.nn.Dropout(), torch.nn.Linear(2*self.emb_dim_rat, 1)),
            self.emb_dim
            )

        self.predictor = torch.nn.Sequential(torch.nn.Linear(3*self.emb_dim, 2*self.emb_dim), torch.nn.BatchNorm1d(2*self.emb_dim,track_running_stats=False), nn_act, torch.nn.Dropout(), torch.nn.Linear(2*self.emb_dim, self.num_classes))
    def forward(self, nodes, edge_indexs, graph_indicators, context_info):
        h_node = self.graph_encoder(nodes, edge_indexs, graph_indicators)
        pos_g, neg_g = self.get_context_emb(context_info)
        h_r, h_env, r_node_num, env_node_num = self.separator(nodes, edge_indexs, graph_indicators, h_node, pos_g, neg_g)
        pos_g = pos_g.repeat(h_r.shape[0],1)
        neg_g = neg_g.repeat(h_r.shape[0],1)
        h_r = torch.cat([h_r, pos_g, neg_g], dim = -1)
        h_env = torch.cat([h_env, pos_g, neg_g], dim = -1)
        h_rep = (h_r.unsqueeze(1) + h_env.unsqueeze(0))
        h_rep = h_rep.view(-1, self.emb_dim*3)
        pred_rem = self.predictor(h_r)
        pred_rep = self.predictor(h_rep)
        loss_reg =  torch.abs(r_node_num / (r_node_num + env_node_num) - self.gamma  * torch.ones_like(r_node_num)).mean()
        output = {'pred_rep': pred_rep, 'pred_rem': pred_rem, 'loss_reg':loss_reg}
        return output
    
    def get_context_emb(self,info):
        nodes, edge_indexs, graph_indicators = info
        h_node = self.graph_encoder(nodes, edge_indexs, graph_indicators)
        batch = graph_indicators
        size = batch[-1].item() + 1 
        h_out_tmp = scatter_add(h_node, batch, dim=0, dim_size=size)
        n_graphs = int((graph_indicators[-1] + 1)/2)
        h_pos_g, h_neg_g = h_out_tmp[:n_graphs].mean([0]), h_out_tmp[n_graphs:].mean([0])
        return h_pos_g, h_neg_g
    
    def get_sparsity(self, nodes, edge_indexs, graph_indicators,context_info):
        h_node = self.graph_encoder(nodes, edge_indexs, graph_indicators)
        pos_g, neg_g = self.get_context_emb(context_info)
        h_r, h_env, r_node_num, env_node_num = self.separator(nodes, edge_indexs, graph_indicators, h_node, pos_g, neg_g)  
        return r_node_num / (r_node_num + env_node_num)

    def get_gate(self, nodes, edge_indexs, graph_indicators,info):
        h_nodes = self.graph_encoder(nodes, edge_indexs, graph_indicators)
        h_pos_g, h_neg_g = self.get_context_emb(info)
        gate = self.separator.get_gate(nodes, edge_indexs, graph_indicators, h_nodes,h_pos_g,h_neg_g)
        return gate




class separator(torch.nn.Module):
    def __init__(self, rationale_gnn_node, gate_nn, emb_dim):
        super(separator, self).__init__()
        self.rationale_gnn_node = rationale_gnn_node
        self.gate_nn = gate_nn
        self.emb_dim = emb_dim

        self.film_beta = torch.nn.Linear(self.emb_dim*2, self.emb_dim)
        self.film_gamma = torch.nn.Linear(self.emb_dim*2, self.emb_dim)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.rationale_gnn_node)
        reset(self.gate_nn)


    
    def forward(self,nodes, edge_indexs, graph_indicators, h_node, h_pos_g, h_neg_g,size=None):
        x = self.rationale_gnn_node(nodes, edge_indexs, graph_indicators)
        batch = graph_indicators
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch[-1].item() + 1 if size is None else size
        # pdb.set_trace()



        h_pos_g = h_pos_g.repeat(x.shape[0], 1)
        h_neg_g = h_neg_g.repeat(x.shape[0], 1)
        x = torch.cat([x, h_pos_g, h_neg_g], dim=-1)


        gate = self.gate_nn(x)

        assert gate.dim() == h_node.dim() and gate.size(0) == h_node.size(0)

        gate = torch.sigmoid(gate)
        # pdb.set_trace()
        h_out = scatter_add(gate * h_node, batch, dim=0, dim_size=size)
        # pdb.set_trace()
        c_out = scatter_add((1 - gate) * h_node, batch, dim=0, dim_size=size)

        r_node_num = scatter_add(gate, batch, dim=0, dim_size=size)
        env_node_num = scatter_add((1 - gate), batch, dim=0, dim_size=size)

        return h_out, c_out, r_node_num + 1e-8 , env_node_num + 1e-8 
    
    def get_gate(self,nodes, edge_indexs, graph_indicators, h_node, h_pos_g, h_neg_g,size=None):
        x = self.rationale_gnn_node(nodes, edge_indexs, graph_indicators)
        batch = graph_indicators
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch[-1].item() + 1 if size is None else size

        h_pos_g = h_pos_g.repeat(x.shape[0], 1)
        h_neg_g = h_neg_g.repeat(x.shape[0], 1)
        x = torch.cat([x, h_pos_g, h_neg_g], dim=-1)
        gate = self.gate_nn(x)
        gate = torch.sigmoid(gate)
        return gate.detach()

