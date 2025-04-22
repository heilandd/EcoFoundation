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
