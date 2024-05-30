import torch
from copy import deepcopy
import torch
import torch.nn.functional as F
import gc
import higher
import itertools
from torch.optim import lr_scheduler
from torch_geometric.nn.inits import reset
import torch.optim as optim
import torch.nn as nn
from model import Model,SupConLoss
nn_act = torch.nn.ReLU()
import pdb
from sklearn.metrics import roc_auc_score
import numpy as np

class Meta(torch.nn.Module):
    def __init__(self, args):
        super(Meta, self).__init__()
        self.args = args
        self.model = Model(args)
        self.device = args.device
        self.dataset = args.dataset
        self.supconloss = SupConLoss(self.device)
        self.global_optim = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.inner_optim = torch.optim.Adam(self.model.predictor.parameters(), lr=args.inner_lr)
        self.lr_scheduler = lr_scheduler.CosineAnnealingLR(self.global_optim, args.epochs*args.train_episode, eta_min=args.lr*0.01)
        self.local_update_step =args.inner_epochs
        self.clip = 5
        self.emb_dim = self.args.emb_dim
        self.num_classes = self.args.way


    def forward(self, support_data, query_data):

        (support_nodes, support_edge_index, support_graph_indicator, support_label,_) = support_data
        (query_nodes, query_edge_index, query_graph_indicator, query_label,_) = query_data
        querysz = query_label.size()[1]

        corrects = []
        stop_gates,scores=[],[]
        train_losses,train_accs=[],[]
        step=0
        loss_q, loss_sup, pred_dic = None, None, None
        with higher.innerloop_ctx(
                self.model, self.inner_optim, copy_initial_weights=False
            ) as (fnet, diffopt):
            for k in range(0, self.local_update_step):
                pred_dic= fnet(support_nodes.squeeze(0), support_edge_index[0], support_graph_indicator.squeeze(0), [support_nodes.squeeze(0), support_edge_index[0], support_graph_indicator.squeeze(0)])


                loss_rem = F.nll_loss(F.log_softmax(pred_dic['pred_rem'].to(torch.float32),dim=1), support_label.squeeze(0))
        

                loss_reg = pred_dic['loss_reg']

                pred_nce = pred_dic['pred_rep'].reshape(support_graph_indicator[0][-1]+1,support_graph_indicator[0][-1]+1,self.num_classes)
                
                loss_nce = self.supconloss(pred_nce,support_label.squeeze(0) )

                loss_sup = loss_rem + loss_reg + loss_nce * self.args.nce_weight_local

                train_losses.append(loss_sup.item())
                diffopt.step(loss_sup)

            with torch.no_grad():
                pred = F.softmax(pred_dic['pred_rem'], dim=1).argmax(dim=1)
                correct = torch.eq(pred, support_label[0]).sum().item()  # convert to numpy
                train_accs.append(correct/support_label[0].size(0))


            pred_dic= fnet(query_nodes.squeeze(0), query_edge_index[0], query_graph_indicator.squeeze(0),[support_nodes.squeeze(0), support_edge_index[0], support_graph_indicator.squeeze(0)])
            loss_rem = F.nll_loss(F.log_softmax(pred_dic['pred_rem'].to(torch.float32),dim=1), query_label.squeeze(0))
            # rep_label = query_label.squeeze(0).repeat_interleave(query_graph_indicator[0][-1]+1,dim=0) 
            # loss_rep = F.nll_loss(F.log_softmax(pred_dic['pred_rep'].to(torch.float32),dim=1), rep_label)
            pred_nce = pred_dic['pred_rep'].reshape(query_graph_indicator[0][-1]+1,query_graph_indicator[0][-1]+1,self.num_classes)
            
            loss_nce = self.supconloss(pred_nce,query_label.squeeze(0) )
    
            
            loss_reg = pred_dic['loss_reg']
            
            loss_q = loss_rem + loss_reg + loss_nce *  self.args.nce_weight_global
            loss_q.backward()
            

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.global_optim.step()
        self.global_optim.zero_grad()
        self.lr_scheduler.step()
        with torch.no_grad():
            pred_q = F.softmax(pred_dic['pred_rem'], dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, query_label[0]).sum().item()  # convert to numpy
            corrects.append(correct)
            
    
        acc_q = corrects[-1] / querysz
        # loss_dict['support_env_loss'] = np.mean(support_env_losses)
        # loss_dict['query_env_loss'] = np.mean(query_env_losses)
        # pdb.set_trace()
        return acc_q * 100,  loss_sup.item(), loss_q.item(), train_accs[-1]

    def finetunning(self, support_data, query_data):

        (support_nodes, support_edge_index, support_graph_indicator, support_label,_) = support_data
        (query_nodes, query_edge_index, query_graph_indicator, query_label,_) = query_data

        querysz = query_label.size()[1]

        # losses_q = [0 for _ in range(self.update_step_test)]  # losses_q[0] is the loss on step 0
        corrects =[]
        step=0
        stop_gates,scores,query_loss=[],[],[]
        with higher.innerloop_ctx(
                self.model, self.inner_optim, copy_initial_weights=False
            ) as (fnet, diffopt):
            for k in range(0, self.local_update_step):
                pred_dic= fnet(support_nodes.squeeze(0), support_edge_index[0], support_graph_indicator.squeeze(0),[support_nodes.squeeze(0), support_edge_index[0], support_graph_indicator.squeeze(0)])
                
                loss_rem = F.nll_loss(F.log_softmax(pred_dic['pred_rem'].to(torch.float32),dim=1), support_label.squeeze(0))
        
                # rep_label = support_label.squeeze(0).repeat_interleave(support_graph_indicator[0][-1]+1,dim=0) 
                # loss_rep = F.nll_loss(F.log_softmax(pred_dic['pred_rep'].to(torch.float32),dim=1), rep_label)
                loss_reg = pred_dic['loss_reg']
                pred_nce = pred_dic['pred_rep'].reshape(support_graph_indicator[0][-1]+1,support_graph_indicator[0][-1]+1,self.num_classes)
                
                loss_nce = self.supconloss(pred_nce,support_label.squeeze(0) )

                loss_sup = loss_rem + loss_reg + loss_nce * self.args.nce_weight_local

                
                diffopt.step(loss_sup)
                

            pred_dic = fnet(query_nodes.squeeze(0), query_edge_index[0], query_graph_indicator.squeeze(0),[support_nodes.squeeze(0), support_edge_index[0], support_graph_indicator.squeeze(0)])
            pred_q = F.softmax(pred_dic['pred_rem'], dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, query_label[0]).sum().item()  # convert to numpy
            corrects.append(correct)
            
            loss_query=F.nll_loss(F.log_softmax(pred_dic['pred_rem'].to(torch.float32),dim=1), query_label.squeeze(0))
            query_loss.append(loss_query.item())
                
        accs = 100 * corrects[-1] / querysz

        return accs,  query_loss[-1], 


    def faith_test(self, support_data, query_data):
        (support_nodes, support_edge_index, support_graph_indicator, support_label,support_gt) = support_data
        (query_nodes, query_edge_index, query_graph_indicator, query_label,query_gt) = query_data
        gates = self.model.get_gate(query_nodes.squeeze(0), query_edge_index[0], query_graph_indicator.squeeze(0),[support_nodes.squeeze(0), support_edge_index[0], support_graph_indicator.squeeze(0)])
        querysz = query_label.size()[1]
        preds, labels = [[] for _ in range(querysz)] ,[[] for _ in range(querysz)]
        for i in range(gates.shape[0]):
            preds[query_graph_indicator[0][i].item()].append(gates[i].item())
            labels[query_graph_indicator[0][i].item()].append(query_gt[0][i].item())
        
        roc_aucs = []
        for i in range(querysz):
            roc_aucs.append(roc_auc_score(np.array(labels[i]), np.array(preds[i])))
        return np.mean(roc_aucs)
    

    
    def get_weights(self, support_data):
        if self.args.dataset=='syn1':
            (support_nodes, support_edge_index, support_graph_indicator, support_label, _) = support_data
        else:
            (support_nodes, support_edge_index, support_graph_indicator, support_label) = support_data
        # pdb.set_trace()
        gates = self.model.get_gate(support_nodes.squeeze(0), support_edge_index, support_graph_indicator.squeeze(0),[support_nodes.squeeze(0), support_edge_index, support_graph_indicator.squeeze(0)])
        return gates