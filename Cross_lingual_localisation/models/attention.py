import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DotProductAttention(nn.Module):
    """ Implementation of the dot product attention as described in https://ieeexplore.ieee.org/abstract/document/9054678.
        Here we use the terminology in the paper: Attention-Based Keyword Localisation in Speech using Visual Grounding
    """

    def __init__(self):
        super(DotProductAttention, self).__init__()

    def forward(self, q_embed, conv_feat):
        """
        Args:
            q_embed: query embeddings. Shape => (vocab_size x emb_dim). Note: embedding dimension is selected to match the dimension of conv_feature
            conv_feat: output of the last convolutional layer of a baseline model architecture (PSC or CNNPool => 
            https://github.com/kayodeolaleye/keyword_localisation_speech)  Shape => (batch_size x emb_dim x input_length)

        Returns:
            attention_weights: batch_size x vocab_size x input_length => used to hypothesize the location of each keyword in the vocabulary for each utterance
            context_vector: batch_size x vocab_size x embed_size => input to the MLP module 
            

        """
        # print("q_emb: ", q_embed.shape)
        # print("conv_feat: ", conv_feat.shape)
        sim_scores = torch.matmul(q_embed, conv_feat)
        attention_weights = F.softmax(sim_scores, dim=2)
        context_vector = torch.matmul(attention_weights, conv_feat.transpose(1, 2))
        # print("context_vector: ", context_vector.shape)
        return context_vector, attention_weights