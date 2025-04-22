################################
#######.   GNN Models   ########
# D. H. Heiland
################################
################################
################################
################################
import os
import torch
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch_geometric.utils as utils
import matplotlib as PL
from tqdm import tqdm
import sklearn
from sklearn import preprocessing
import matplotlib.pyplot as plt
from torch_geometric.nn import global_mean_pool
import torch
from torch.nn import BatchNorm1d, Linear
from torch_geometric.nn import GATConv
from torch_geometric.loader import DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch.nn import Linear
import torch.optim as optim
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GATConv
import torch
import torch.nn as nn
import torch
from torch_geometric.nn import GATConv, global_mean_pool, LayerNorm
from torch.nn import Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", device)

############ GAN  ####################
class CustomGATConv(GATConv):
    def __init__(self, *args, **kwargs):
        super(CustomGATConv, self).__init__(*args, **kwargs)
        self.attention_weights = None

    def forward(self, x, edge_index, size=None, return_attention_weights=False):
        # Original forward method
        out = super().forward(x, edge_index, size)

        # Store attention weights
        if return_attention_weights:
            self.attention_weights = self._alpha
            return out, self.attention_weights
        else:
            return out
class Graph_MERFISH(torch.nn.Module):
    def __init__(self, num_features_exp, hidden_channels, num_classes):
        super(Graph_MERFISH, self).__init__()

        # Attention GAT Conv Layers
        per_head_hidden_channels = hidden_channels // 5
        self.conv1_exp = GATConv(num_features_exp, per_head_hidden_channels, heads=5)
        self.conv2_exp = GATConv(per_head_hidden_channels * 5, per_head_hidden_channels, heads=5)


        # Batch norm layers
        self.bn1 = torch.nn.LayerNorm(hidden_channels)
        self.bn2 = torch.nn.LayerNorm(hidden_channels)
        self.dropout = torch.nn.Dropout(0.5) # Add dropout for regularization

        # Latent space
        self.merge = Linear(hidden_channels, hidden_channels)
        torch.nn.init.xavier_uniform_(self.merge.weight.data)

        # MLP Prediction Status
        self.mlp_Recurent = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5), # Add dropout in the MLP as well
            torch.nn.Linear(hidden_channels, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, data):
        exp, edge_index = data.x, data.edge_index

        

        # GATConv layers require edge_index to be long type
        x_exp = exp
        edge_index = edge_index.long()

        x_exp, attention_weights_1 = self.conv1_exp(x_exp, edge_index, return_attention_weights=True)
        x_exp = F.leaky_relu(x_exp)
        x_exp = self.dropout(self.bn1(x_exp))

        x_exp, attention_weights_2 = self.conv2_exp(x_exp, edge_index, return_attention_weights=True)
        x_exp = F.leaky_relu(x_exp)
        x_exp = self.dropout(self.bn2(x_exp))

        x = self.merge(x_exp)
        x = F.leaky_relu(x)

        Recurrent_out = self.mlp_Recurent(global_mean_pool(x, data.batch))

        return x, Recurrent_out, attention_weights_1, attention_weights_2

## Early Stopping
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        if val_loss < self.val_loss_min:
            self.val_loss_min = val_loss
            # Save the model
            torch.save(model.state_dict(), 'checkpoint.pt')

############ GAN -no- Discriminator  ####################    
## Train model without discriminator
def train_withoutdiscriminator(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    early_stopping = EarlyStopping(patience=patience, delta=0.01)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        total_discriminator_loss = 0

        for data in tqdm(train_loader):
            # Train primary model
            optimizer.zero_grad()
            latent, Recurrent_out, AT1, AT2 = model(data.to(device))
            gt = data.Class.long().to(device)
            primary_loss = criterion(Recurrent_out, gt)
            primary_loss.backward()
            optimizer.step()
            epoch_loss += primary_loss.item()

        # Validation (primary model only)
        model.eval()
        val_loss = 0
        val_outputs = []
        val_labels = []

        with torch.no_grad():
            for data in val_loader:
                latent, Recurrent_out, AT1, AT2 = model(data.to(device))
                val_outputs.append(torch.argmax(Recurrent_out, dim=1).detach().cpu().numpy())
                gt = data.Class.long().to(device)
                val_labels.append(gt.cpu().numpy())
                val_loss += criterion(Recurrent_out, gt).item()

        val_loss /= len(val_loader)
        pred = np.concatenate(val_outputs)
        true_labels = np.concatenate(val_labels)

        # Print results
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("Train Loss: {:.4f}, Discriminator Loss: {:.4f}, Val Loss: {:.4f}".format(epoch_loss / len(train_loader), total_discriminator_loss / len(train_loader), val_loss))
        print("Validation Accuracy:", accuracy_score(true_labels, pred))

        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    model.load_state_dict(torch.load('checkpoint.pt'))













