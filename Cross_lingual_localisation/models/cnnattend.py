import random
import numpy as np
import torch
import torch.nn as nn
from models.attention import DotProductAttention

class CNNAttend(nn.Module):

    def __init__(self, vocab_size, embed_size, fc_layer_size, dropout):
        super(CNNAttend, self).__init__()

        # Convolutional module
        self.conv_module = nn.Sequential(
                nn.Conv1d(39, 96, 9, 1, 4),
                nn.ReLU(),
                # nn.BatchNorm1d(96, affine=False),
                nn.Conv1d(96, 96, 11, 1, 5),
                nn.ReLU(),
                # nn.BatchNorm1d(96, affine=False),
                nn.Conv1d(96, 96, 11, 1, 5),
                nn.ReLU(),
                # nn.BatchNorm1d(96, affine=False),
                nn.Conv1d(96, 96, 11, 1, 5),
                nn.ReLU(),
                # nn.BatchNorm1d(96, affine=False),
                nn.Conv1d(96, 96, 11, 1, 5),
                nn.ReLU(),
                # nn.BatchNorm1d(96, affine=False),
                nn.Conv1d(96, embed_size, 11, 1, 5)
                # nn.ReLU()
            )

        # Embedding module
        self.embed = embed_queries(embed_size, vocab_size)
        
        # Attention module
        self.attention_module = DotProductAttention()
        
        # MLP module
        self.mlp_module = nn.Sequential(
            nn.Linear(embed_size, fc_layer_size),
            nn.ReLU(),
            nn.Linear(fc_layer_size, 1),
            nn.Dropout(p=dropout)
    )

    def forward(self, x):
     
        conv_feat = self.conv_module(x)
        context_vector, attention_weights = self.attention_module(self.embed.cuda(), conv_feat)
        output = self.mlp_module(context_vector).squeeze()

        return output, attention_weights

def embed_queries(embed_size, vocab_size):
    q_embed = torch.zeros(vocab_size, embed_size)
    embeddings = nn.Embedding(vocab_size, embed_size)
    for i in range(vocab_size):
        lookup_tensor = torch.tensor([i], dtype=torch.long)
        embed = embeddings(lookup_tensor)
        
        q_embed[i, :] = embed
        
    return q_embed
