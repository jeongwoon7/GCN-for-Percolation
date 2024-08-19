""""
This module implements a Graph Convolutional Network (GCN) model
for percolation classification tasks using custom datasets.
The module defines the necessary classes and functions for
data handling, model architecture, training, and evaluation.
"""


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from time import time
import os, psutil
from torch.optim.lr_scheduler import StepLR

# module variables
bs = 20
num_rows = 10
num_cols = 10
node_dim = 2
npcs = 21 # divisor
pcs = np.linspace(0.0, 1.0, npcs)

# Dimension of data
input_dim = 2  # Dimension of node feature vectors
hidden_dim = 2 # Hidden dimension of the CGN layer
output_dim = 2  # Output dimension of the GCN layer
in_channel = output_dim
num_classes = 2  # percolation or not-percolation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


"""------------------------------------------------------------------------
1. Define custom dataset and labeling according to the confusion method
------------------------------------------------------------------------"""

class MyDataset(Dataset):
    """
    Custom dataset class for loading data.

    Args:
        df (pd.DataFrame): DataFrame containing the graph data.
        num_rows (int): Number of rows in the lattice.
        num_cols (int): Number of columns in the lattice.

    Attributes:
        df (pd.DataFrame): Stores the DataFrame.
        num_rows (int): Number of rows in the lattice.
        num_cols (int): Number of columns in the lattice.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Retrieves the sample at the specified index.
    """

    def __init__(self, df, num_rows, num_cols, device):
        self.df = df
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.device = device

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        p = self.df.loc[idx, 'p']
        lattice = np.fromstring(self.df.loc[idx, 'lattice'][1:-1], dtype=int, sep=',').reshape(num_rows, num_cols)
        adjacency_matrix = np.fromstring(self.df.loc[idx, 'adjacency_matrix'][1:-1], dtype=int, sep=',').reshape(
            num_rows * num_cols, num_rows * num_cols)
        node_features = np.array(eval(self.df.loc[idx, 'node_features']))

        p = torch.tensor(p, dtype=torch.float32).to(self.device)
        lattice = torch.tensor(lattice, dtype=torch.float32).to(self.device)
        adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32).to(self.device)
        node_features = torch.tensor(node_features, dtype=torch.float32).to(self.device)


        sample = {
            'p': p,
            'lattice': lattice,
            'adjacency_matrix': adjacency_matrix,
            'node_features': node_features
        }
        # node feature
        # If at the top boundary,    (occ_status,20-epochs_from_pc0.6)
        # If at the bottom boundary, (occ_status,20-epochs_from_pc0.55_2nd)
        # Otherwise,                 (occ_status,0)
        return sample


def set_label(p_batch,pc):
    """
    Assign labels using the confusion method as described in the main text.

    Args:
        p_batch (torch.Tensor): Batch of occupation probabilities p used for sample generation.
        pc (float): Parameterized percolation threshold pc'.

    Returns:
        torch.Tensor: Boolean tensor indicating if transition occurs.
    """

    bool_confusion = p_batch > pc  # if True, transition
    if pc == 1.0:  # When not specifically set, the confusion label becomes False instead of True at the boundary.
        bool_confusion = p_batch >= pc

    return bool_confusion


class BatchData:
    """
    Helper class to generate batch data for training.

    Args:
        data (dict): Batch data containing features, adjacency matrix, and labels.

    Attributes:
        p (torch.Tensor): Site occupation probabilities.
        lattice (torch.Tensor): Lattice configurations.
        adj (torch.Tensor): Adjacency matrices.
        x (torch.Tensor): Node features.
    """

    def __init__(self,data):
        self.p = torch.as_tensor(data['p'], dtype=torch.float32).to(device)
        self.lattice = torch.as_tensor(data['lattice'], dtype=torch.float32).to(device)
        self.adj = torch.as_tensor(data['adjacency_matrix'], dtype=torch.float32).to(device)
        self.x = torch.as_tensor(data['node_features'], dtype=torch.float32).to(device)


"""------------------------------------------------------------------------
2. Proposed model architecture
------------------------------------------------------------------------"""

class my_GCN(torch.nn.Module):
    """
    Graph Convolutional Network (GCN) with configurable depth.
    Recommended setting for lattice with the number of rows = M  : n_layers = int((M+1)/2)

    Args:
        input_dim (int): Dimension of input node features.
        hidden_dim (int): Hidden dimension of the GCN layer.
        output_dim (int): Output dimension of the GCN layer.
        n_layers (int): Number of GCN layers.

    Attributes:
        fcs (torch.nn.ModuleList): List of fully connected layers.
        output_fc (nn.Linear): Output layer of the GCN.

    Methods:
        forward(x, adj): Performs the forward pass through the GCN.
    """


    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(my_GCN, self).__init__()
        self.fcs = torch.nn.ModuleList()

        # input layer
        self.fcs.append(nn.Linear(input_dim, hidden_dim))

        # add hidden layers, which conserve the size of hidden channels
        for i in range(n_layers-1):
            self.fcs.append(nn.Linear(hidden_dim, hidden_dim))

        # output layer
        self.output_fc = nn.Linear(hidden_dim, output_dim)

    def forward(self,x,adj):
        for fc in self.fcs:
            x0 = x.clone()
            x = fc(x)
            x = torch.bmm(adj, x)
            x = x0 +  F.relu(x)

        x = self.output_fc(x)
        x = torch.sigmoid(x)
        return x



def coarse_graining(A):
    """
    Coarse-grain pooling to convert GCN output to a tensor of size (batch size, in_channel, 5, 5).

    Args:
        A (torch.Tensor): GCN output tensor.

    Returns:
        torch.Tensor: Coarse-grained tensor.
    """

    batch_tmp = A.shape[0]
    original_array = A.reshape(batch_tmp, num_rows, num_cols, in_channel)

    if original_array.shape[1]==10:
        cg_factor=2
        new_shape = original_array.shape[1] // cg_factor
        block_array = original_array.reshape(batch_tmp, new_shape, cg_factor, new_shape, cg_factor, in_channel)
        block_means = block_array.mean(axis=(2, 4))

    else:
        # step 1. First, make the matrix of size 10x10
        cg_factor_0 = num_rows // 10  # 2 for 20, 3 for 30 ...
        new_shape = original_array.shape[1] // cg_factor_0
        block_array = original_array.reshape(batch_tmp, new_shape, cg_factor_0, new_shape, cg_factor_0, in_channel)
        block_means = block_array.mean(axis=(2, 4))

        # step 2. Convert 10x10 matrix into 5x5
        cg_factor = 2
        new_array= block_means
        new_shape = new_array.shape[1] // cg_factor
        block_array = new_array.reshape(batch_tmp, new_shape, cg_factor, new_shape, cg_factor, in_channel)
        block_means = block_array.mean(axis=(2, 4))


    reshaped = block_means.permute(0, 3, 1, 2)

    return reshaped  # torch tensor of size (bs, in_channel, 5, 5)



class SimpleCNN_softmax(nn.Module):

    """
    Convolutional Neural Network (CNN) for processing coarse-grained outputs from GCN.

    Args:
        in_channel (int): Number of input channels.
        num_classes (int): Number of output classes.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        pool (nn.MaxPool2d): Max pooling layer.
        conv2 (nn.Conv2d): Second convolutional layer.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.

    Methods:
        forward(x): Performs the forward pass through the CNN.
    """

    def __init__(self, in_channels, num_classes):
        super(SimpleCNN_softmax, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 1 * 1, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 32 * 1 * 1)  # Adjust the size based on the final feature map size (1x1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)

        # Apply softmax activation to the final layer's output for binary classification
        x = F.softmax(x,dim=1)
        return x


# Combined model for GCN and CNN
class CombinedModel(nn.Module):

    def __init__(self, gcn_model, cnn_model):
        super(CombinedModel, self).__init__()
        self.gcn_model = gcn_model
        self.cnn_model = cnn_model

    def forward(self, x_gcn, adjacency_matrix):
        # x_gcn : original node features

        # Forward pass through the GCN model
        gcn_output = self.gcn_model(x_gcn, adjacency_matrix)

        # reshape the M x M matrix into 5 x 5 coarse grained matrix
        reshaped_block_matrix = coarse_graining(gcn_output)

        # Forward pass through the cnn model
        x = self.cnn_model(reshaped_block_matrix)
        return x


"""------------------------------------------------------------------------
3. Functions for training and evaluation of the machine learning model
------------------------------------------------------------------------"""
# Train loop for model training
def train_loop(train_dataloader, pc, combined_model, loss_fn, optimizer):

    num_batches = len(train_dataloader)
    train_loss = 0

    label_true = torch.tensor([1.0, 0.0], device=device)  # transition
    label_false = torch.tensor([0.0, 1.0], device=device)  # no transition

    for batch_data in train_dataloader:

        # Extract the batch data
        data = BatchData(batch_data)
        p_batch = data.p
        adjacency_matrix_batch = data.adj
        node_features_batch = data.x

        output_batch = combined_model(node_features_batch, adjacency_matrix_batch)


        # Assign labels based on the prediction of model
        bool_prediction = output_batch[:, 0] > output_batch[:, 1]  # True : transition

        # Assign labels by using confusion method (purposefully mislabelling)
        bool_confusion = set_label(p_batch, pc)

        # Construct label tensor for batch learning
        # Assign [1, 0] when the bool is true, and assign [0, 1] when the bool is false
        label_prediction = torch.where(bool_prediction.unsqueeze(1), label_true, label_false)
        label_confusion = torch.where(bool_confusion.unsqueeze(1), label_true, label_false)

        # Minimize the difference between output_batch and the label made from confusion method
        loss = loss_fn(label_confusion, output_batch)
        train_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print the loss for each epoch
    train_loss /= num_batches

    return train_loss


# Test loop for validation, i.e., calculating overall and p-resolved performances
"""------------------------------------------------------------------------
# Option for identifying p of each configuration
if option=True, read p used for generating the configuration
if option=False, calculate occupied site and p as following : p = (occupied sites/number of nodes)
------------------------------------------------------------------------"""
@torch.no_grad()
def test_loop(test_dataloader, pc, combined_model, option=True):

    num_total_config = len(test_dataloader.dataset)
    count_true = 0
    p_for_each = []
    result_for_each=[]

    for batch_data in test_dataloader:

        # Extract the batch data
        data = BatchData(batch_data)
        p_batch = data.p
        lattice_batch = data.lattice
        adjacency_matrix_batch = data.adj
        node_features_batch = data.x

        output_batch = combined_model(node_features_batch, adjacency_matrix_batch)

        # Compute the performance
        bool_prediction = output_batch[:, 0] > output_batch[:, 1]
        bool_confusion = set_label(p_batch, pc)

        bool_list = bool_prediction == bool_confusion

        if option == True:
            p_for_each.extend(p_batch.tolist())
        else :
            count_occupied_sites=lattice_batch.sum(dim=(1,2))
            num_nodes = lattice_batch.size(1)*lattice_batch.size(2)
            p = count_occupied_sites/num_nodes
            p_for_each.extend(p.tolist())

        result_for_each.extend(bool_list.tolist())

        count = torch.sum(bool_list)
        count_true += count.item()

    performance = count_true / num_total_config
    print(f'{pc:>4.2f} performance = {performance:>12.8f}')

    return performance, p_for_each, result_for_each


# Define train and evaluate loop for parallel computation
def train_and_evaluate(num_runs, pc, bs, lr, epochs, loss_diff_criterion,max_epochs_without_improvement, weight_decay, step_size, gamma):

    a_list = []
    performance_list = []

    for run in range(num_runs):

        print('--------------------------')
        print(f'{run}-th run, start pc = {pc:>4.2f}')
        print()

        # Confusion method
        df_train = pd.read_csv('./configurations.csv')
        df_test = pd.read_csv('./configurations_test.csv')

        # Print process ID and get the CPU number on which the current process is running
        process = psutil.Process(os.getpid())
        cpu_num = process.cpu_num()
        print(f"Process ID: {os.getpid()}, CPU Core: {cpu_num}")

        random_seed = np.random.randint(0, 10000, size=1)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        # Generate a unique identifier for the process
        unique_key = f"pc_{pc}_pid_{os.getpid()}_cpu_{cpu_num}"

        # Dataloader
        train_dataset = MyDataset(df_train, num_rows, num_cols,device)
        test_dataset = MyDataset(df_test, num_rows, num_cols,device)
        data_loaders = {}
        data_loaders[unique_key] = {
            "train_dataloader": DataLoader(train_dataset, batch_size=bs, shuffle=True),
            "test_dataloader": DataLoader(test_dataset, batch_size=bs, shuffle=False)
        }

        # Models and parameters
        n_layer = 5
        models = {}
        gcn_model = my_GCN(input_dim, hidden_dim, output_dim, n_layer)
        cnn_model = SimpleCNN_softmax(in_channel, num_classes)
        combined_model = CombinedModel(gcn_model, cnn_model)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(combined_model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

        models[unique_key] = {
            "gcn_model": gcn_model,
            "cnn_model": cnn_model,
            "combined_model": combined_model,
            "loss_fn": loss_fn,
            "optimizer": optimizer,
            "scheduler": scheduler
        }

        previous_loss = float('inf')
        epochs_without_improvement = 0

        memory_info = psutil.virtual_memory()
        print(f"Total memory: {memory_info.total}")
        print(f"Available memory: {memory_info.available}")
        print(f"Used memory: {memory_info.used}")
        print(f"Memory percentage: {memory_info.percent}%")


        for epoch in range(epochs):

            tic = time()
            loss = train_loop(data_loaders[unique_key]["train_dataloader"], pc, models[unique_key]["combined_model"], models[unique_key]["loss_fn"], models[unique_key]["optimizer"])

            # Adjust learning rate
            models[unique_key]["scheduler"].step()

            # Check if the loss difference is below the criterion
            loss_diff = abs(loss - previous_loss)

            if loss_diff > loss_diff_criterion:
                # Reset the count of consecutive epochs without improvement
                epochs_without_improvement = 0
            else:
                # Increment the count of consecutive epochs without improvement
                epochs_without_improvement += 1

            # Check if the count exceeds the maximum threshold
            if epochs_without_improvement > max_epochs_without_improvement:
                break  # Exit the loop

            # Update the previous loss for the next epoch
            previous_loss = loss

            toc = time()

            if epoch == 0:
                print(f"pc={pc:4.2f}, time elapsed={toc - tic}")
                print()

            if epoch == 0 or (epoch + 1) % 10 == 0:
                print(f'    pc={pc:>4.2f}       Epoch={epoch + 1}, loss={loss:>16.8f}')

        a, p_for_each, result_for_each = test_loop(data_loaders[unique_key]["test_dataloader"], pc, models[unique_key]["combined_model"])

        # Set the number of grids in y-axis for the 2D performance plot
        ngrid = 21
        grid = np.linspace(0, 1, ngrid)
        counts = [0] * (ngrid - 1)
        denominator = [0] * (ngrid - 1)

        for value, flag in zip(p_for_each, result_for_each):
            # changed to count denominator, and to add to counts[] only when the label is True.
            if value < grid[-2]:
                for j in range(ngrid - 2):
                    if grid[j] <= value < grid[j + 1]:
                        denominator[j] += 1
                        if flag:
                            counts[j] += 1
            else:
                denominator[-1] += 1
                if flag:
                    counts[-1] += 1

        performance_ratio = np.array(counts) / np.array(denominator)


        # Temporary DataFrames to store the results of this run
        a_list.append(a)
        performance_list.append(performance_ratio)

        # Save the model
        fname="pc_"+ str(pc)+".pt"
        PATH=os.path.join("./saved_models",fname)
        torch.save(models[unique_key]["combined_model"].state_dict(),PATH)

    mean_a = np.mean(a_list)
    std_a = np.std(a_list)
    mean_performance_ratio = sum(performance_list) / len(performance_list)

    return pc, mean_a, std_a, mean_performance_ratio



"""------------------------------------------------------------------------
4. Implementation of Differential pooling method and corresponding training/validating

Most train_diff and test_diff are just slightly modified versions of train_loop and test_loop for implementing DiffPool.
------------------------------------------------------------------------"""

# Basic GNN model used for DiffPool
class GNN(torch.nn.Module):
    def __init__(self,in_channels,hidden_channels,out_channels,n_layers):
        super(GNN,self).__init__()

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()

        # input layer
        self.convs.append(GCNConv(in_channels,hidden_channels))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.dropouts.append(torch.nn.Dropout(0.1))

        # add hidden layers, which conserve the size of hidden channels
        for i in range(n_layers-1):
            self.convs.append(GCNConv(hidden_channels,hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            self.dropouts.append(torch.nn.Dropout(0.1))

        # output layer
        self.convs.append(GCNConv(hidden_channels,out_channels))
        self.bns.append(torch.nn.BatchNorm1d(out_channels))
        self.dropouts.append(torch.nn.Dropout(0.1))

    def forward(self,x,adj,mask=None):
        for conv,bn,dropout in zip(self.convs,self.bns,self.dropouts):
            x=conv(x,adj)
            x = x.transpose(1,2)
            x = bn(x)
            x = x.transpose(1,2)
            x = F.relu(x)
            x = dropout(x)

        x = torch.sigmoid(x)
        return x


class DiffPool(torch.nn.Module):
    def __init__(self, hidden_dim, n_layers):
        super(DiffPool, self).__init__()

        num_cluster1 = ceil(0.25 * max_nodes)
        num_cluster2 = ceil(0.25 * 0.25 * max_nodes)

        # hidden channel is a constant of the model in this case
        self.gnn1_embed = GNN(input_dim, hidden_dim,hidden_dim,n_layers)
        self.gnn2_embed = GNN(hidden_dim,hidden_dim,hidden_dim,n_layers)
        self.gnn3_embed = GNN(hidden_dim,hidden_dim,hidden_dim,n_layers)

        self.gnn1_pool = GNN(input_dim, hidden_dim,num_cluster1,n_layers)
        self.gnn2_pool = GNN(hidden_dim,hidden_dim,num_cluster2,n_layers)

        self.lin1 = torch.nn.Linear(hidden_dim,hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim,output_dim)


    def forward(self,x0,adj0):
        s0 = self.gnn1_pool(x0,adj0)
        z0 = self.gnn1_embed(x0,adj0)
        x1,adj1,l1,e1 = dense_diff_pool(z0,adj0,s0)

        s1 = self.gnn2_pool(x1,adj1)
        z1 = self.gnn2_embed(x1,adj1)
        x2, adj2, l2, e2 = dense_diff_pool(z1, adj1, s1)

        z2 = self.gnn3_embed(x2,adj2)

        graph_vec = z2.mean(dim=1)
        graph_vec = F.relu(self.lin1(graph_vec))
        graph_vec = self.lin2(graph_vec)

        return F.log_softmax(graph_vec, dim= -1), l1+l2, e1+e2


def train_diff(loader,pc,model,optimizer):

    model.train()
    loss_all = 0

    for batch_data in loader:
        data = BatchData(batch_data)
        p_batch = data.p
        # resets the gradients of all optimized parameters to zero
        # preparing the system for a new gradient computation
        optimizer.zero_grad()
        output, _,_ = model(data.x,data.adj)

        bool_confusion = set_label(p_batch,pc)
        y = torch.where(bool_confusion.unsqueeze(1), 1, 0)

        loss = F.nll_loss(output,y.view(-1))
        loss.backward()
        loss_all += y.size(0) * loss.item()
        optimizer.step()

    return loss_all / len(loader.dataset)


@torch.no_grad() #Disable gradient calculation of test()
def test_diff(loader,pc,model,option=True):
    model.eval()
    correct = 0
    p_for_each = []
    result_for_each = []

    for data in loader:
        data = BatchData(data)
        p_batch = data.p
        lattice_batch = data.lattice
        bool_confusion = set_label(p_batch, pc)
        y = torch.where(bool_confusion.unsqueeze(1), 1, 0)

        # output = model(data.x,data.adj)[0] : output of the model (bs,1)
        # output.max(dim=1) finds the maximum values in each row
        # and return the values and the corresponding indices.
        output = model(data.x,data.adj)[0]

        pred = output.max(dim=1)[1] # index of the maximum value. ex) transition :(0.9,0.1) -> it will gives 0
        correct += pred.eq(y.view(-1)).sum().item()

        # for data analysis
        bool_prediction = output[:, 0] < output[:, 1]
        bool_list = bool_prediction == bool_confusion


        if option == True:
            p_for_each.extend(p_batch.tolist())
        else :
            count_occupied_sites=lattice_batch.sum(dim=(1,2))
            num_nodes = lattice_batch.size(1)*lattice_batch.size(2)
            p = count_occupied_sites/num_nodes
            p_for_each.extend(p.tolist())

        result_for_each.extend(bool_list.tolist())

    performance = correct / len(loader.dataset)
    print(f'{pc:>4.2f} performance = {performance:>12.8f}')

    return performance, p_for_each, result_for_each


def train_and_evaluate_diff(num_runs, pc, bs,lr,epochs,loss_diff_criterion,max_epochs_without_improvement):

    a_list = []
    performance_list = []

    for run in range(num_runs):


        print('--------------------------')
        print(f'{run}-th run, start pc = {pc:>4.2f}')
        print()

        # Confusion method
        df_train = pd.read_csv('./configurations.csv')
        df_test = pd.read_csv('./configurations_test.csv')

        # Print process ID and get the CPU number on which the current process is running
        process = psutil.Process(os.getpid())
        cpu_num = process.cpu_num()
        print(f"Process ID: {os.getpid()}, CPU Core: {cpu_num}")

        random_seed = np.random.randint(0, 10000, size=1)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        # Generate a unique identifier for the process
        unique_key = f"pc_{pc}_pid_{os.getpid()}_cpu_{cpu_num}"

        # Dataloader
        train_dataset = MyDataset(df_train, num_rows, num_cols, device)
        test_dataset = MyDataset(df_test, num_rows, num_cols, device)
        data_loaders = {}
        data_loaders[unique_key] = {
            "train_dataloader": DataLoader(train_dataset, batch_size=bs, shuffle=True),
            "test_dataloader": DataLoader(test_dataset, batch_size=bs, shuffle=False)
        }

        # Models and parameters
        models = {}
        model = DiffPool(hidden_dim, n_layers).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        models[unique_key] = {
            "model": model,
            "optimizer": optimizer
        }

        memory_info = psutil.virtual_memory()
        print(f"Total memory: {memory_info.total}")
        print(f"Available memory: {memory_info.available}")
        print(f"Used memory: {memory_info.used}")
        print(f"Memory percentage: {memory_info.percent}%")

        previous_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(epochs):
            tic = time()
            loss = train_diff(data_loaders[unique_key]["train_dataloader"], pc, models[unique_key]["model"], models[unique_key]["optimizer"])

            # Check if the loss difference is below the criterion
            loss_diff = abs(loss - previous_loss)

            if loss_diff > loss_diff_criterion:
                # Reset the count of consecutive epochs without improvement
                epochs_without_improvement = 0
            else:
                # Increment the count of consecutive epochs without improvement
                epochs_without_improvement += 1

            # Check if the count exceeds the maximum threshold
            if epochs_without_improvement > max_epochs_without_improvement:
                break  # Exit the loop

            # Update the previous loss for the next epoch
            previous_loss = loss

            toc = time()

            if epoch == 0:
                print(f"pc={pc:4.2f}, time elapsed={toc - tic}")
                print()

            if epoch == 0 or (epoch + 1) % 10 == 0:
                print(f'    pc={pc:>4.2f}       Epoch={epoch + 1}, loss={loss:>16.8f}')

        a, p_for_each, result_for_each = test_diff(data_loaders[unique_key]["test_dataloader"],pc, models[unique_key]["model"])

        # Set the number of grids in y-axis for the 2D performance plot
        ngrid = 21
        grid = np.linspace(0, 1, ngrid)
        counts = [0] * (ngrid - 1)
        denominator = [0] * (ngrid - 1)

        for value, flag in zip(p_for_each, result_for_each):

            if value < grid[-2]:
                for j in range(ngrid - 2):
                    if grid[j] <= value < grid[j + 1]:
                        denominator[j] += 1
                        if flag:
                            counts[j] += 1
            else:
                denominator[-1] += 1
                if flag:
                    counts[-1] += 1

        performance_ratio = np.array(counts) / np.array(denominator)

        # Temporary DataFrames to store the results of this run
        a_list.append(a)
        performance_list.append(performance_ratio)

        # Save the model
        fname="pc_"+ str(pc)+".pt"
        PATH=os.path.join("./saved_models",fname)
        torch.save(models[unique_key]["model"].state_dict(),PATH)

    mean_a = np.mean(a_list)
    std_a = np.std(a_list)
    mean_performance_ratio = sum(performance_list) / len(performance_list)

    return pc, mean_a, std_a, mean_performance_ratio
