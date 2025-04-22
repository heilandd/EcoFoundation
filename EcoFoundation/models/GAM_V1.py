"""
GAM_V1.py

Implements a graph attention-based encoder with optional adversarial domain adaptation,
prediction heads, and early stopping for training convergence.

Dependencies:
- torch
- torch_geometric
"""

## Imports:
import torch
from torch_geometric.nn import GATConv
from torch_geometric.utils import softmax
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

## Graph Attention Layer
class CustomGATConv(GATConv):
    def __init__(self, in_channels, out_channels, heads=1, concat=True, edge_dim=1, **kwargs):
        super().__init__(in_channels, out_channels, heads=heads, concat=concat, edge_dim=edge_dim, **kwargs)

    def forward(self, x, edge_index, edge_attr=None, return_attention_weights=False):
        # Fix for edge_attr shape issue after batching
        if edge_attr is not None and edge_attr.dim() == 2 and edge_attr.shape[1] == 1:
            edge_attr = edge_attr.squeeze(1)

        if return_attention_weights:
            out, attn_weights = super().forward(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
            return out, attn_weights
        else:
            return super().forward(x, edge_index, edge_attr=edge_attr)

## Graph Encoder:
class GraphEncoder(torch.nn.Module):
    def __init__(self, num_features_exp, hidden_channels, edge_dim=1):
        super(GraphEncoder, self).__init__()
        per_head_hidden = hidden_channels // 5
        self.conv1 = CustomGATConv(num_features_exp, per_head_hidden, heads=5, edge_dim=edge_dim)
        self.conv2 = CustomGATConv(per_head_hidden * 5, per_head_hidden, heads=5, edge_dim=edge_dim)
        self.bn1 = torch.nn.LayerNorm(hidden_channels)
        self.bn2 = torch.nn.LayerNorm(hidden_channels)
        self.dropout = torch.nn.Dropout(0.5)
        self.merge = torch.nn.Linear(hidden_channels, hidden_channels)
        torch.nn.init.xavier_uniform_(self.merge.weight.data)

    def forward(self, x, edge_index, edge_attr, batch):
        x, att1 = self.conv1(x, edge_index, edge_attr, return_attention_weights=True)
        x = F.leaky_relu(x)
        x = self.dropout(self.bn1(x))
        x, att2 = self.conv2(x, edge_index, edge_attr, return_attention_weights=True)
        x = F.leaky_relu(x)
        x = self.dropout(self.bn2(x))
        x = self.merge(x)
        x = F.leaky_relu(x)
        
        ## Reduce latentspace on subgraph level
        x_pooled = global_mean_pool(x, batch)

        return x, x_pooled, att1, att2

## Prediction heads: 
class MLP(torch.nn.Module):
    def __init__(self, hidden_channels, num_classes):
        super(MLP, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden_channels, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.model.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)


## if required for batch effect removal:

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)  

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# EarlyStopping:
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

