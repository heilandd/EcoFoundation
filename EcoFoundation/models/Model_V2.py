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

def edge_index_to_adj(edge_index, num_nodes):
    """
    Convert edge_index to an adjacency matrix.
    :param edge_index: Tensor of shape [2, num_edges] containing the graph edges.
    :param num_nodes: Number of nodes in the graph.
    :return: Dense adjacency matrix A of shape [num_nodes, num_nodes].
    """
    # Create the adjacency matrix
    adj_matrix = to_dense_adj(edge_index, max_num_nodes=num_nodes)
    
    # Remove the additional dimension introduced by to_dense_adj
    adj_matrix = adj_matrix.squeeze(0)
    
    # Ensure the adjacency matrix has the correct dimensions
    if adj_matrix.dim() == 2:
        adj_matrix = adj_matrix.unsqueeze(0)  # Make it [1, num_nodes, num_nodes] for consistency
    
    return adj_matrix

class GTN(nn.Module):
    def __init__(self, num_edge, num_channels, w_in, w_out, num_class, num_layers, norm):
        super(GTN, self).__init__()
        self.num_edge = num_edge
        self.num_channels = num_channels
        self.w_in = w_in
        self.w_out = w_out
        self.num_class = num_class
        self.num_layers = num_layers
        self.is_norm = norm
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(GTLayer(num_edge, num_channels, first=True))
            else:
                layers.append(GTLayer(num_edge, num_channels, first=False))
        self.layers = nn.ModuleList(layers)
        self.weight = nn.Parameter(torch.Tensor(w_in, w_out))
        self.bias = nn.Parameter(torch.Tensor(w_out))
        self.linear1 = nn.Linear(self.w_out * self.num_channels, self.w_out)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def gcn_conv(self, X, H):
        X = torch.mm(X, self.weight)
        H = self.norm(H, add=True)
        return torch.mm(H.t(), X)

    def normalization(self, H):
        for i in range(self.num_channels):
            if i == 0:
                H_ = self.norm(H[i, :, :]).unsqueeze(0)
            else:
                H_ = torch.cat((H_, self.norm(H[i, :, :]).unsqueeze(0)), dim=0)
        return H_

    def norm(self, H, add=False):
        H = H.t()
        if not add:
            H = H * ((torch.eye(H.shape[0]) == 0).type(torch.FloatTensor))
        else:
            H = H * ((torch.eye(H.shape[0]) == 0).type(torch.FloatTensor)) + torch.eye(H.shape[0]).type(torch.FloatTensor)
        deg = torch.sum(H, dim=1)
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0
        deg_inv = deg_inv * torch.eye(H.shape[0]).type(torch.FloatTensor)
        H = torch.mm(deg_inv, H)
        H = H.t()
        return H

    def forward(self, A, X, target_x=None, target=None):
        Ws = []
        for i in range(self.num_layers):
            if i == 0:
                H, W = self.layers[i](A)
            else:
                H = self.normalization(H)
                H, W = self.layers[i](A, H)
            Ws.append(W)

        for i in range(self.num_channels):
            if i == 0:
                X_ = F.relu(self.gcn_conv(X, H[i]))
            else:
                X_tmp = F.relu(self.gcn_conv(X, H[i]))
                X_ = torch.cat((X_, X_tmp), dim=1)
        X_ = self.linear1(X_)
        X_ = F.relu(X_)

        # Remove or comment out the classification part since there's no target
        # y = self.linear2(X_[target_x])
        # loss = self.loss(y, target)
        # return loss, y, Ws

        return X_, Ws  # Return the embeddings and the learned weights

class GTLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, first=True):
        super(GTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first
        if self.first == True:
            self.conv1 = GTConv(in_channels, out_channels)
            self.conv2 = GTConv(in_channels, out_channels)
        else:
            self.conv1 = GTConv(in_channels, out_channels)
    
    def forward(self, A, H_=None):
        if self.first == True:
            a = self.conv1(A)
            b = self.conv2(A)
            H = torch.bmm(a,b)
            W = [(F.softmax(self.conv1.weight, dim=1)).detach(),(F.softmax(self.conv2.weight, dim=1)).detach()]
        else:
            a = self.conv1(A)
            H = torch.bmm(H_,a)
            W = [(F.softmax(self.conv1.weight, dim=1)).detach()]
        return H,W

class GTConv(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(GTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(out_channels,in_channels,1,1))
        self.bias = None
        self.scale = nn.Parameter(torch.Tensor([0.1]), requires_grad=False)
        self.reset_parameters()
    def reset_parameters(self):
        n = self.in_channels
        nn.init.constant_(self.weight, 0.1)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, A):
        A = torch.sum(A*F.softmax(self.weight, dim=1), dim=1)
        return A

class GeneDecoder(torch.nn.Module):
    def __init__(self, hidden_channels, num_features_exp):
        super(GeneDecoder, self).__init__()
        self.fc1 = torch.nn.Linear(hidden_channels, hidden_channels*2)
        self.fc2 = torch.nn.Linear(hidden_channels*2, num_features_exp)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class GraphTransformer(torch.nn.Module):
    def __init__(self, num_features_exp, hidden_channels, nr_heads, num_layers, num_edge, num_channels, norm=True):
        super(GraphTransformer, self).__init__()

        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.num_channels = num_channels

        # GTN-specific initialization
        self.gtn = GTN(num_edge=num_edge, num_channels=num_channels, w_in=num_features_exp, w_out=hidden_channels, num_class=hidden_channels, num_layers=num_layers, norm=norm)

        # Adjust this part to match the previous output feature size
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
        X, edge_index = data.x, data.edge_index

        # Convert edge_index to adjacency matrix
        A = edge_index_to_adj(edge_index, data.num_nodes)

        # GTN forward pass
        X_, Ws = self.gtn(A, X, None, None)  # No target, just get embeddings
        
        # Latent space embedding
        x = self.merge(X_)  # Ensure this matches the output features from the GTN
        x = F.sigmoid(x)

        # Reconstruct the original feature space
        reconstructed_x = self.decoder(x)

        return x, reconstructed_x, Ws

# Contrastive loss function (NT-Xent loss)
class NTXentLoss(torch.nn.Module):
    def __init__(self, temperature=1.0):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        
    def forward(self, z_i, z_j):
        z_i = F.normalize(z_i, dim=-1)
        z_j = F.normalize(z_j, dim=-1)
        sim = self.cosine_similarity(z_i.unsqueeze(1), z_j.unsqueeze(0)) / self.temperature
        
        # Diagonal entries (self-similarity) should be excluded from softmax
        logits = torch.cat([sim, sim.T], dim=0)
        labels = torch.arange(z_i.size(0), device=sim.device)
        labels = torch.cat([labels, labels], dim=0)
        
        # Contrastive loss with cross entropy
        loss = F.cross_entropy(logits, labels)
        return loss

class SimCLRLoss(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    def forward(self, z_i, z_j):
        N = z_i.size(0)
        z = torch.cat((z_i, z_j), dim=0)
        sim = self.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        labels = torch.cat([torch.arange(N) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        mask = torch.eye(labels.shape[0], dtype=torch.bool)
        labels = labels[~mask].view(labels.shape[0], -1)
        sim = sim[~mask].view(sim.shape[0], -1)

        loss = -torch.log((labels * torch.exp(sim)).sum(1) / torch.exp(sim).sum(1)).mean()
        return loss

# Masking function to create a student graph by masking nodes
def mask_nodes(data, mask_ratio=0.5):
    num_nodes = data.num_nodes
    mask = torch.rand(num_nodes, device=device) < mask_ratio
    masked_data = data.clone()
    masked_data.x[mask] = 0  
    return masked_data, mask

# Function to train the model using batches from a DataLoader
def train_model(model, train_loader, epochs=100, learning_rate=1e-3, temperature=0.5, alpha=0.5, mask_ratio=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    contrastive_loss_fn = NTXentLoss(temperature=temperature).to(device)
    reconstruction_loss_fn = torch.nn.MSELoss().to(device)

    model.train()
    loss_values = []  # List to store combined loss values

    for epoch in range(epochs):
        total_loss = 0
        for data in tqdm(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            # Forward pass: Get latent embeddings and reconstructed features
            latent_embeddings, reconstructed_x, _ = model(data)
            
            # Masked graph (student graph)
            masked_data, mask = mask_nodes(data, mask_ratio=mask_ratio)
            masked_latent_embeddings, masked_reconstructed_x, _ = model(masked_data)
            
            # Contrastive loss 
            z_teacher = latent_embeddings
            z_student = masked_latent_embeddings
            contrastive_loss = contrastive_loss_fn(z_teacher, z_student)
            
            # Reconstruction loss (between original and reconstructed features)
            reconstruction_loss = reconstruction_loss_fn(reconstructed_x, data.x)
            
            # Combine the losses
            loss = alpha * contrastive_loss + (1 - alpha) * reconstruction_loss
            total_loss += loss.item()

            # Backpropagation
            loss.backward()
            optimizer.step()

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        loss_values.append(avg_loss)

        print(f'Epoch {epoch}, Loss: {avg_loss}')

    return loss_values
