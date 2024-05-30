This code is released for the paper: Towards Few-shot Self-explaining Graph Neural Networks

Dependencies:
pytorch, higher, pytorch_geometric, scikit-learn, 

First, download MNIST data and transform with mnist.ipynb

Then, generate validation and test data by generate_val_test_data.py

Finally, to train and test MSE-GNN, run main.py with args following this training example:

python main.py --dataset mnist --gnn gin --device 2 --lr 0.00001 --inner_lr 0.001 --seed 2024 --gamma 0.1 --shot 5


