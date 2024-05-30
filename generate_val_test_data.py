import argparse
from dataset import GraphDataSet, FewShotDataloader, get_dataset
from tqdm import tqdm
from utils import *
from args import *
import pickle
from datetime import datetime
import higher
from tqdm import tqdm
import random
import numpy as np
import pdb
from torch.nn import functional as F
from models.new.model import *
import random
import time
args = get_args()
datasets = ['mnist']
set_seed(1)
for dataset in datasets:
    for shot in [3,5,10]:
        args.dataset = dataset
        validation_set = GraphDataSet("val", "mnist")
        test_set = GraphDataSet("test", "mnist")
        val_loader = FewShotDataloader(validation_set,
                                        n_way=2,
                                        n_shot=5,
                                        n_query=10,
                                        batch_size=1,
                                        num_workers=0,
                                        epoch_size=100,
                                        )
        test_loader = FewShotDataloader(test_set,
                                n_way=2,  # number of novel categories.
                                n_shot=shot,  # number of training examples per novel category.
                                n_query=10,  # number of test examples for all the novel categories.
                                batch_size=1,  # number of training episodes per batch.
                                num_workers=0,
                                epoch_size=100,  # number of batches per epoch.
                                )
        test_data = []
        for i, data in enumerate(tqdm(test_loader(1)), 1):
            test_data.append(data)

        val_data = []
        for i, data in enumerate(tqdm(val_loader(1)), 1):
            val_data.append(data)
        output = open('../datasets/' + dataset + '/2way_' + str(shot) +'shot_10query_100episode_test_data.pkl', 'wb')
        pickle.dump(test_data,output)
        output.close()

        output = open('../datasets/' + dataset + '/2way_' + str(shot) +'shot_10query_100episode_val_data.pkl', 'wb')
        pickle.dump(val_data,output)
        output.close()

