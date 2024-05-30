import os
import os.path as osp
import shutil
import pickle
import torch.utils.data as data
import numpy as np
import torch
import random
import torchnet as tnt
import pdb
from collections import defaultdict
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from ogb.graphproppred import PygGraphPropPredDataset
class GraphDataSet(data.Dataset):
    def __init__(self,phase="val", dataset_name="mnist"):
        super(GraphDataSet,self).__init__()
        self.datasetName=dataset_name
        self.phase=phase

        self.base_folder="../datasets/" + dataset_name
        if dataset_name in "mnist":
            self.num_features=3
            self.classes = range(10)
        elif dataset_name in 'cifar10':
            self.num_features=3
            self.classes = range(10)
        self.data_list = torch.load((self.base_folder+"/{}.pt".format(phase)))
        self.num_graph=len(self.data_list)
        self.label2graphs = defaultdict(list)
        for data in self.data_list:
            self.label2graphs[data.y.item()].append(data)
    def __getitem__(self, index):
        return self.data_list[index]
    def __len__(self):
        return len(self.data_list)


class FewShotDataloader_syn():
    def __init__(self,
                 dataset = "syn",
                 n_way=2,
                 n_shot=2,
                 n_query=2,
                 batch_size=1,
                 num_workers=0,
                 epoch_size=200,
                 phase='train'
                 ):
        self.dataset = dataset
        self.phase = phase
        self.n_shot = n_shot
        self.n_way = n_way
        self.n_query = n_query
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epoch_size = epoch_size
        self.dataset = torch.load('../datasets/' + self.dataset + '_dataset.pt')
        self.train_classes = ["grid","fan","clique","cycle","diamond"]
        self.val_classes = ["crane","triangle"]
        self.test_classes = ["house","star","wheel"]
        self.is_eval_mode=(self.phase=='test') or (self.phase=='val')
        self.class_int = ["cycle", "grid", "diamond","star","fan","clique","crane","house","wheel","triangle"]

        
    def sample_graph(self, sampled_classes):
        support_graphs, query_graphs = [], []
        support_node_attr, support_edge_index1, support_edge_index2, support_labels, support_graph_indicator = [], [], [], [], []
        query_node_attr, query_edge_index1, query_edge_index2, query_labels, query_graph_indicator = [], [], [], [], []
        support_node_gt, query_node_gt = [], []
        for label in sampled_classes:
            graphs=self.dataset["tree"][label] + self.dataset["ba"][label]
            assert (len(graphs)) >= self.n_shot + self.n_query
            selected_graphs=random.sample(graphs,self.n_shot+self.n_query)
            support_graphs.extend(selected_graphs[:self.n_shot])
            query_graphs.extend(selected_graphs[self.n_shot:])
        
        


        for i, g in enumerate(support_graphs):
            n_id_0 = len(support_node_attr)
            n_nodes = g.feat.shape[0]
            support_node_attr.extend(g.feat.tolist())
            support_edge_index1.extend((g.edge_index[0] + n_id_0).tolist())
            support_edge_index2.extend((g.edge_index[1] + n_id_0).tolist())
            support_labels.append(sampled_classes.index(self.class_int[g.y.item()]))
            support_graph_indicator.extend([i]*n_nodes)
            support_node_gt.extend(g.node_label[0].tolist())


            
        
        for i, g in enumerate(query_graphs):
            n_id_0 = len(query_node_attr)
            n_nodes = g.feat.shape[0]
            query_node_attr.extend(g.feat.tolist())
            query_edge_index1.extend((g.edge_index[0] + n_id_0).tolist())
            query_edge_index2.extend((g.edge_index[1] + n_id_0).tolist())
            query_labels.append(sampled_classes.index(self.class_int[g.y.item()]))
            query_graph_indicator.extend([i]*n_nodes)
            query_node_gt.extend(g.node_label[0].tolist())


        support_edge_index = [support_edge_index1, support_edge_index2]
        query_edge_index = [query_edge_index1, query_edge_index2]
        return [torch.from_numpy(np.array(support_node_attr)).float(), torch.from_numpy(np.array(support_edge_index)).long(), torch.from_numpy(np.array(support_graph_indicator)).long(), torch.from_numpy(np.array(support_labels)).long(),torch.from_numpy(np.array(support_node_gt)).long()], \
                [torch.from_numpy(np.array(query_node_attr)).float(), torch.from_numpy(np.array(query_edge_index)).long(), torch.from_numpy(np.array(query_graph_indicator)).long(), torch.from_numpy(np.array(query_labels)).long(),torch.from_numpy(np.array(query_node_gt)).long()]
    


    def sample_episode(self):
        if self.phase == "train":
            classes= random.sample(self.train_classes,self.n_way)
            support_data,query_data=self.sample_graph(classes)
            return support_data,query_data
        
        elif self.phase == 'test':
            classes= random.sample(self.test_classes,self.n_way)
            support_data,query_data=self.sample_graph(classes)
            return support_data,query_data
        
        else:
            classes= random.sample(self.val_classes,self.n_way)
            print(classes)
            support_data,query_data=self.sample_graph(classes)
            return support_data,query_data


    def get_iterator(self, epoch=0):
            rand_seed = epoch
            random.seed(rand_seed)
            np.random.seed(rand_seed)
            def load_function(iter_idx):
                support_data,query_data =self.sample_episode()
                return support_data,query_data
            tnt_dataset = tnt.dataset.ListDataset(
                elem_list=range(self.epoch_size), load=load_function)
            data_loader = tnt_dataset.parallel(
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=(False if self.is_eval_mode else True)
                # shuffle=True
            )

            return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return int(self.epoch_size / self.batch_size)



class FewShotDataloader():
    def __init__(self,
                 dataset,
                 n_way=2, # number of novel categories.
                 n_shot=2, # number of training examples per novel category.
                 n_query=2, # number of test examples for all the novel categories.
                 batch_size=1, # number of training episodes per batch.
                 num_workers=0,
                 epoch_size=2000, # number of batches per epoch.
                 ):
        self.dataset = dataset
        self.data_list = dataset.data_list
        self.label2graphs = dataset.label2graphs
        self.phase = self.dataset.phase
        # max_possible_nKnovel = (self.dataset.num_cats_base if self.phase=='train'
        #                         else self.dataset.num_cats_novel)
        self.n_way=n_way
        self.n_shot=n_shot
        self.n_query=n_query

        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.num_workers = num_workers
        self.is_eval_mode=(self.phase=='test') or (self.phase=='val')



    def sample_graph(self, sampled_classes):
        support_graphs, query_graphs = [], []
        support_node_attr, support_edge_index1, support_edge_index2, support_labels, support_graph_indicator = [], [], [], [], []
        query_node_attr, query_edge_index1, query_edge_index2, query_labels, query_graph_indicator = [], [], [], [], []
        support_node_gt, query_node_gt = [], []
        for label in sampled_classes:
            graphs=self.label2graphs[label]
            assert (len(graphs)) >= self.n_shot + self.n_query
            selected_graphs=random.sample(graphs,self.n_shot+self.n_query)
            support_graphs.extend(selected_graphs[:self.n_shot])
            query_graphs.extend(selected_graphs[self.n_shot:])
        
        


        for i, g in enumerate(support_graphs):
            n_id_0 = len(support_node_attr)
            n_nodes = g.x.shape[0]
            support_node_attr.extend(g.x.tolist())
            support_edge_index1.extend((g.edge_index[0] + n_id_0).tolist())
            support_edge_index2.extend((g.edge_index[1] + n_id_0).tolist())
            support_labels.append(sampled_classes.index(g.y.item()))
            support_graph_indicator.extend([i]*n_nodes)
            support_node_gt.extend(g.label_ex.squeeze().tolist())

            
        
        for i, g in enumerate(query_graphs):
            n_id_0 = len(query_node_attr)
            n_nodes = g.x.shape[0]
            query_node_attr.extend(g.x.tolist())
            query_edge_index1.extend((g.edge_index[0] + n_id_0).tolist())
            query_edge_index2.extend((g.edge_index[1] + n_id_0).tolist())
            query_labels.append(sampled_classes.index(g.y.item()))
            query_graph_indicator.extend([i]*n_nodes)
            query_node_gt.extend(g.label_ex.squeeze().tolist())

        support_edge_index = [support_edge_index1, support_edge_index2]
        query_edge_index = [query_edge_index1, query_edge_index2]

        return [torch.from_numpy(np.array(support_node_attr)).float(), torch.from_numpy(np.array(support_edge_index)).long(), torch.from_numpy(np.array(support_graph_indicator)).long(), torch.from_numpy(np.array(support_labels)).long(),torch.from_numpy(np.array(support_node_gt)).long()], \
                [torch.from_numpy(np.array(query_node_attr)).float(), torch.from_numpy(np.array(query_edge_index)).long(), torch.from_numpy(np.array(query_graph_indicator)).long(), torch.from_numpy(np.array(query_labels)).long(),torch.from_numpy(np.array(query_node_gt)).long()]
    
        





    def sample_episode(self):
        classes= random.sample(self.label2graphs.keys(),self.n_way)
        support_data,query_data=self.sample_graph(classes)
        return support_data,query_data

    def get_iterator(self, epoch=0):
        rand_seed = epoch
        random.seed(rand_seed)
        np.random.seed(rand_seed)
        def load_function(iter_idx):
            support_data,query_data =self.sample_episode()
            return support_data,query_data
        tnt_dataset = tnt.dataset.ListDataset(
            elem_list=range(self.epoch_size), load=load_function)
        data_loader = tnt_dataset.parallel(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=(False if self.is_eval_mode else True)
            # shuffle=True
        )

        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return int(self.epoch_size / self.batch_size)

def get_dataset(args):
    if 'syn' in args.dataset:
        train_data = None
        test_data_path = '../datasets/' + args.dataset + '/' + str(args.way) + 'way_' + str(args.shot) + 'shot_' + str(args.query) + 'query_' + str(args.val_episode) + 'episode_test_data.pkl' 
        val_data_path = '../datasets/' + args.dataset + '/' + str(args.way) + 'way_' + str(args.shot) + 'shot_' + str(args.query) + 'query_' + str(args.val_episode) + 'episode_val_data.pkl' 
        f = open(test_data_path, 'rb')
        test_data = pickle.load(f)
        f.close()
        
        f = open(val_data_path, 'rb')
        val_data = pickle.load(f)
        f.close()

    else:
        train_data=GraphDataSet(phase="train",dataset_name=args.dataset)
        test_data_path = '../datasets/' + args.dataset + '/' + str(args.way) + 'way_' + str(args.shot) + 'shot_' + str(args.query) + 'query_' + str(args.val_episode) + 'episode_test_data.pkl' 
        val_data_path = '../datasets/' + args.dataset + '/' + str(args.way) + 'way_' + str(args.shot) + 'shot_' + str(args.query) + 'query_' + str(args.val_episode) + 'episode_val_data.pkl' 
        f = open(test_data_path, 'rb')
        test_data = pickle.load(f)
        f.close()
        f = open(val_data_path, 'rb')
        val_data = pickle.load(f)
        f.close()
    return train_data,val_data,test_data
