import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import DataLoader
from torch.nn import Linear
from tqdm import tqdm
import matplotlib.pyplot as plt

# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

class Graph_MERFISH(torch.nn.Module):
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
            latent_embeddings, reconstructed_x, _, _ = model(data)
            
            # Masked graph (student graph)
            masked_data, mask = mask_nodes(data, mask_ratio=mask_ratio)
            masked_latent_embeddings, masked_reconstructed_x, _, _ = model(masked_data)
            
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
