import argparse
def get_args():
    parser = argparse.ArgumentParser(description='MSE-GNN')
    parser.add_argument('--device', type=int, default=1,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin',
                        help='gin or gcn')
    parser.add_argument('--drop_ratio', type=float, default=0.3, 
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=2,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=128,
                        help='dimensionality of hidden units in GNNs (default: 128)')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='size ratio to regularize the explanation (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--inner_epochs', type=int, default=5,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-2)')
    parser.add_argument('--inner_lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-2)')                    
    parser.add_argument('--dataset', type=str, default="mnist")
    parser.add_argument('--way', type=int, default=2)
    parser.add_argument('--train_episode', type=int, default=200)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=10)
    parser.add_argument('--val_episode', type=int, default=100)
    parser.add_argument('--num_features', type=int, default=3)
    parser.add_argument('--seed' ,type=int,default=1, help='random seed')
    parser.add_argument('--tune', action='store_true', default=False)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--model', type=str)
    parser.add_argument('--nce_weight_local', type=float, default=0.1)
    parser.add_argument('--nce_weight_global', type=float, default=0.1)
    args = parser.parse_args()
    return args
