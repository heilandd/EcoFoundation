import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import DataLoader
from torch.nn import Linear
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch_geometric.utils import to_dense_adj
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from matplotlib import pyplot as plt
import pdb
from torch_geometric.nn import PNAConv
from torch_geometric.nn import CGConv
import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class EcoGAT(torch.nn.Module):
    def __init__(self, num_features_exp, hidden_channels, nr_heads):
        super(Graph_MERFISH, self).__init__()

        # Attention GAT Conv Layers
        per_head_hidden_channels = hidden_channels // nr_heads
        self.conv1_exp = GATConv(num_features_exp, per_head_hidden_channels, heads=nr_heads)
        self.conv2_exp = GATConv(hidden_channels, per_head_hidden_channels, heads=nr_heads)  

        # Batch norm layers
        self.bn1 = torch.nn.LayerNorm(hidden_channels)
        self.bn2 = torch.nn.LayerNorm(hidden_channels)
        self.dropout = torch.nn.Dropout(0.5)  

        # Latent space
        self.merge = Linear(hidden_channels, hidden_channels)
        torch.nn.init.xavier_uniform_(self.merge.weight.data)

        # Decoder layer to reconstruct the original feature space
        self.decoder = GeneDecoder(hidden_channels, num_features_exp)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, data):
        exp, edge_index = data.x, data.edge_index

        x_exp = exp
        edge_index = edge_index.long()

        x_exp, attention_weights_1 = self.conv1_exp(x_exp, edge_index, return_attention_weights=True)
        x_exp = F.leaky_relu(x_exp)
        x_exp = self.dropout(self.bn1(x_exp))

        x_exp, attention_weights_2 = self.conv2_exp(x_exp, edge_index, return_attention_weights=True)
        x_exp = F.leaky_relu(x_exp)
        x_exp = self.dropout(self.bn2(x_exp))

        # Latent space embedding
        x = self.merge(x_exp)
        x = F.sigmoid(x_exp)

        # Reconstruct the original feature space
        reconstructed_x = self.decoder(x)

        return x, reconstructed_x, attention_weights_1, attention_weights_2

### Encoder with edge features ####

class EcoNENN(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_channels):
        super(NENN, self).__init__()
        self.node_embedding = torch.nn.Linear(num_node_features, hidden_channels)
        self.edge_embedding = torch.nn.Linear(num_edge_features, hidden_channels)
        self.conv1 = GATConv(hidden_channels, hidden_channels, heads=4, edge_dim=hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels, heads=4, edge_dim=hidden_channels)

        self.bn1 = torch.nn.LayerNorm(hidden_channels)
        self.bn2 = torch.nn.LayerNorm(hidden_channels)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Embedding node and edge features
        x = F.relu(self.node_embedding(x))
        edge_attr = F.relu(self.edge_embedding(edge_attr))

        # First convolution
        x, attn_weights1 = self.conv1(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        x = self.bn1(F.leaky_relu(x))
        x = self.dropout(x)

        # Second convolution
        x, attn_weights2 = self.conv2(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        x = self.bn2(F.leaky_relu(x))
        x = self.dropout(x)

        return x, attn_weights1, attn_weights2

class EcoPNAGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_channels, aggregators=['mean', 'min', 'max', 'std']):
        super(PNAGNN, self).__init__()
        self.node_embedding = torch.nn.Linear(num_node_features, hidden_channels)
        self.edge_embedding = torch.nn.Linear(num_edge_features, hidden_channels)
        self.conv1 = PNAConv(in_channels=hidden_channels, out_channels=hidden_channels, aggregators=aggregators, edge_dim=hidden_channels)
        self.conv2 = PNAConv(in_channels=hidden_channels, out_channels=hidden_channels, aggregators=aggregators, edge_dim=hidden_channels)

        self.bn1 = torch.nn.LayerNorm(hidden_channels)
        self.bn2 = torch.nn.LayerNorm(hidden_channels)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Embedding node and edge features
        x = F.relu(self.node_embedding(x))
        edge_attr = F.relu(self.edge_embedding(edge_attr))

        # First convolution
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(F.leaky_relu(x))
        x = self.dropout(x)

        # Second convolution
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.bn2(F.leaky_relu(x))
        x = self.dropout(x)

        return x

class EcoCGCNN(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_channels):
        super(CGCNN, self).__init__()
        self.node_embedding = torch.nn.Linear(num_node_features, hidden_channels)
        self.edge_embedding = torch.nn.Linear(num_edge_features, hidden_channels)
        self.conv1 = CGConv(hidden_channels, edge_dim=hidden_channels)
        self.conv2 = CGConv(hidden_channels, edge_dim=hidden_channels)

        self.bn1 = torch.nn.LayerNorm(hidden_channels)
        self.bn2 = torch.nn.LayerNorm(hidden_channels)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Embedding node and edge features
        x = F.relu(self.node_embedding(x))
        edge_attr = F.relu(self.edge_embedding(edge_attr))

        # First convolution
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(F.leaky_relu(x))
        x = self.dropout(x)

        # Second convolution
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(F.leaky_relu(x))
        x = self.dropout(x)

        return x

## EGNN "Exploiting Edge Features in Graph Neural Networks" ####
class EGNNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim, aggr='add'):
        super(EGNNConv, self).__init__(aggr=aggr)
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.edge_lin = torch.nn.Linear(edge_dim, out_channels)
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin.weight)
        torch.nn.init.xavier_uniform_(self.edge_lin.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_attr):
        # Add self-loops to the adjacency matrix
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Linearly transform node and edge features
        x = self.lin(x)
        edge_attr = self.edge_lin(edge_attr)

        # Start propagating messages
        return self.propagate(edge_index, x=x, edge_attr=edge_attr) + self.bias

    def message(self, x_j, edge_attr):
        # Compute messages by combining node and edge features
        return x_j + edge_attr

    def update(self, aggr_out):
        # Apply non-linearity
        return F.relu(aggr_out)
class EcoEGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_channels):
        super(EGNN, self).__init__()
        self.conv1 = EGNNConv(num_node_features, hidden_channels, num_edge_features)
        self.conv2 = EGNNConv(hidden_channels, hidden_channels, num_edge_features)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # First convolutional layer
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = self.dropout(x)

        # Second convolutional layer
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = self.dropout(x)

        return x
