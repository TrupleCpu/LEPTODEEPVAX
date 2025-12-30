import torch
import torch.nn as nn
import torch.nn.functional as F

class BioFeatureEngineer(nn.Module):
    """
    CUSTOM BIOLOGICAL ENCODER (Built from Scratch)
    """
    def __init__(self, vocab_size=25, embed_dim=128, hidden_dim=320):
        super(BioFeatureEngineer, self).__init__()
        
        # Digital Dictionary: Mapping AA strings to Integers
        self.amino_to_idx = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY-X")}
        
        # Embedding Layer: Learns physicochemical properties of each AA
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional Encoding: Allows the AI to understand the sequence order
        self.pos_encoding = nn.Parameter(torch.zeros(1, 1000, embed_dim))
        
        # Attention Block: Learns long-range dependencies between amino acids
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        # Convolutional Feature Extractors (Motif Finders)
        self.conv_block = nn.Sequential(
            nn.Conv1d(embed_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, sequence):
        # Convert string to indices
        indices = [self.amino_to_idx.get(aa, 21) for aa in sequence]
        x = torch.LongTensor(indices).unsqueeze(0).to(next(self.parameters()).device)
        
        # Apply Embedding + Positional Context
        x = self.embedding(x)
        x = x + self.pos_encoding[:, :x.size(1), :]
        
        # Self-Attention Mechanism (The 'From Scratch' Transformer Logic)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attention = torch.matmul(q, k.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        attention = F.softmax(attention, dim=-1)
        x = torch.matmul(attention, v)
        
        # Spatial Feature Extraction
        x = x.transpose(1, 2) # Prep for Conv1d
        x = self.conv_block(x)
        
        # Global Feature Compression
        x = self.pool(x).squeeze(-1)
        return x # Returns 320-dim vector