import argparse
from dataset import GraphDataSet, FewShotDataloader, get_dataset, FewShotDataloader_syn
from tqdm import tqdm
from utils import *
from args import *
from datetime import datetime

import numpy as np
from copy import deepcopy
from torch.nn import functional as F
from meta import Meta
from model import Model
import time
from shutil import copyfile
import os
from utils import *
args = get_args()

t = time.time()
max_val_acc=0
set_seed(args.seed)
datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")
if args.tune:
    args.result_path = './tune_result/model'
else:
    args.result_path = './new-result/' + str(datetime_now) + args.dataset + "gnn" + str(args.gnn) + "lr" +str(args.lr) + "inner_lr" + str(args.inner_lr) + "shot" + str(args.shot) + "inner_epoch" + str(args.inner_epochs) + "gamma" + str(args.gamma) +"g" + str(args.nce_weight_global)+ "l" + str(args.nce_weight_local)+ "seed" + str(args.seed)
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    copyfile('./model.py', args.result_path + '/model.py')
    copyfile('./meta.py', args.result_path + '/meta.py')
    copyfile('./encoder.py', args.result_path + '/encoder.py')
    copyfile('./args.py', args.result_path + '/args.py')
    copyfile('./main.py', args.result_path + '/main.py')

log_path = args.result_path + '/_log.log'
logger = Logger.init_logger(filename=log_path)
logger.info(args)
device = torch.device('cuda:%d' % args.device if torch.cuda.is_available() else 'cpu')
args.device = device
# device = torch.device('cpu')



if 'syn' in args.dataset:
    args.num_features = 10
    _, val_data, test_data = get_dataset(args)
    train_loader = FewShotDataloader_syn(
                                    dataset=args.dataset,
                                    n_way=args.way,
                                    n_shot=args.shot,
                                    n_query=args.query,
                                    batch_size=1,  # number of training episodes per batch.
                                    num_workers=0,
                                    epoch_size=args.train_episode,
                                    phase="train"
                                    )
                
                                

else:
    training_set, val_data, test_data = get_dataset(args)
    train_loader = FewShotDataloader(training_set,
                                    n_way=args.way,
                                    n_shot=args.shot,
                                    n_query=args.query,
                                    batch_size=1,  # number of training episodes per batch.
                                    num_workers=0,
                                    epoch_size=args.train_episode  # number of batches per epoch.
                                    )

meta_model=Meta(args).to(device)
flag=False
max_acc_val = 0.0
early_stop_cnt = 0
for epoch in range(args.epochs):
    if early_stop_cnt >= 20:
        break
    loss_train = 0.0
    correct = 0
    meta_model.train()
    sup_accs, query_accs, sup_losses, query_losses, val_accs, val_losses = [], [], [], [],[], []

    # other loss


    test_accs = []
    sparsity, fidelity = [], []
    for i, data in enumerate(train_loader(epoch), 1):
        support_data, query_data=data
        support_data=[item.to(device) for item in support_data]
        query_data=[item.to(device) for item in query_data]
        acc_q, loss_sup, loss_q, acc_sup = meta_model(support_data, query_data)
        sup_losses.append(loss_sup)
        query_losses.append(loss_q)
        sup_accs.append(acc_sup)
        query_accs.append(acc_q)

        
        if (i+1)%100==0:
            logger.info("acc_sup{:.6f},acc_que{:.6f}, sup_loss{:.6f}, query_loss{:.6f}".format(np.mean(sup_accs),np.mean(query_accs), np.mean(sup_losses), np.mean(query_losses))) 
            

    meta_model.eval()
    for i, data in enumerate(val_data):
        support_data, query_data=data
        support_data=[item.to(device) for item in support_data]
        query_data=[item.to(device) for item in query_data]
        acc,  query_losses_val = meta_model.finetunning(support_data, query_data)
        val_accs.append(acc)

    val_acc_avg=np.mean(val_accs)
    train_acc_avg=np.mean(query_accs)
    
    train_loss_avg =np.mean(query_losses)

    val_acc_ci95 = 1.96 * np.std(np.array(val_accs)) / np.sqrt(args.val_episode)
    
    if val_acc_avg > max_val_acc:
        early_stop_cnt = 0
        max_val_acc = val_acc_avg
        logger.info('\nEpoch(***Best***): {:04d},loss_train: {:.6f},acc_train: {:.6f},'
                                'acc_val:{:.2f} ±{:.2f},time: {:.2f}s,best {:.2f}'
            .format(epoch,train_loss_avg,train_acc_avg,val_acc_avg,val_acc_ci95, time.time() - t,max_val_acc))
        model_filename = args.result_path +'/' + str(args.seed)+ '.pth'
        if os.path.exists(model_filename):
            os.remove(model_filename)
        torch.save({'epoch': epoch, 'embedding':meta_model.state_dict(),
                    # 'optimizer': optimizer.state_dict()
                    }, model_filename)
        meta_model.eval()
        faith_scores = []
        for i, data in enumerate(test_data):
            support_data, query_data = data
            support_data = [item.to(device) for item in support_data]
            query_data = [item.to(device) for item in query_data]
            acc, query_losses= meta_model.finetunning(support_data, query_data)
            faith_score = meta_model.faith_test(support_data, query_data)
            faith_scores.append(faith_score)
            test_accs.append(acc)
        test_acc_avg=np.mean(test_accs)
        faith_score_avg = np.mean(faith_scores)
        test_acc_ci95 = 1.96 * np.std(np.array(test_accs)) / np.sqrt(args.val_episode)
        logger.info('\nacc_test:{:.2f} ±{:.2f},time: {:.2f}s'.format(test_acc_avg,test_acc_ci95,time.time() - t))
        logger.info("faith_score:{:.6f}".format(faith_score_avg))
        

    else :
        early_stop_cnt += 1
        logger.info('\nEpoch: {:04d},loss_train: {:.6f},acc_train: {:.6f},'
                                'acc_val:{:.2f} ±{:.2f},time: {:.2f}s,best {:.2f}'
            .format(epoch, train_loss_avg, train_acc_avg, val_acc_avg, val_acc_ci95, time.time() - t, max_val_acc))


logger.info('Optimization Finished! Total time elapsed: {:.6f}'.format(time.time() - t))


