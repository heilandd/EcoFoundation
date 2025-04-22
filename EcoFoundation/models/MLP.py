
## MLP Head
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import DataLoader
from torch.nn import Linear
from tqdm import tqdm
import matplotlib.pyplot as plt


class MLPHead(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPHead, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class GraphMERFISHWithMLP(torch.nn.Module):
    def __init__(self, pretrained_model, mlp_head):
        super(GraphMERFISHWithMLP, self).__init__()
        self.pretrained_model = pretrained_model
        self.mlp_head = mlp_head

    def forward(self, data):
        z, _, _, _ = self.pretrained_model(data)
        pooled_z = global_mean_pool(z, data.batch)
        #print(f'Pooled Z shape: {pooled_z.shape}')
        output = self.mlp_head(pooled_z)
        return output


def train_mlp_head(model, train_loader, loss_fn, epochs=100, learning_rate=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(model.mlp_head.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data in tqdm(train_loader):
            optimizer.zero_grad()
            
            data = data.to(device)
            gt = data.Class.long()

            # Forward pass
            output = model(data)
            loss = loss_fn(output, gt)

            # Backpropagation
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {avg_loss}')






