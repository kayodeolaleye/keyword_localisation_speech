import torch.nn as nn
import torch
import torch.nn.functional as F
from models.attention import DotProductAttention

class CNNPoolAttend(nn.Module):

    def __init__(self, vocab_size, embed_size):
  
        super(CNNPoolAttend, self).__init__()

        # Convolutional module
        self.conv_module = nn.Sequential(
            nn.Conv1d(39, 64, 9, 1, 4),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(64, 256, 11, 1, 5),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(256, embed_size, 11, 1, 5)
            # nn.ReLU()
        )
   
        # Embedding module
        self.embed = embed_queries(embed_size, vocab_size)

        # Attention module
        self.attention_module = DotProductAttention()
        
        # MLP module
        self.mlp_module = nn.Sequential(
            nn.Linear(embed_size, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1),
            nn.Dropout(p=0.0)
        )

    def forward(self, x):
        conv_feat = self.conv_module(x)
        context_vector, attention_weights = self.attention_module(self.embed.cuda(), conv_feat)
        output = self.mlp_module(context_vector).squeeze()
        return output, attention_weights


def embed_queries(embed_size, vocab_size):
    # torch.manual_seed(1)
    # np.random.seed(1)
    q_embed = torch.zeros(vocab_size, embed_size)
    embeddings = nn.Embedding(vocab_size, embed_size)
    for i in range(vocab_size):
        lookup_tensor = torch.tensor([i], dtype=torch.long)
        embed = embeddings(lookup_tensor)
        
        q_embed[i, :] = embed
        
    return q_embed
