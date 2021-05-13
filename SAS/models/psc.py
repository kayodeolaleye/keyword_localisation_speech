import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from config import input_dim

class PSC(nn.Module):
    """
    This class defines a ConvNet architecture with a structure similar to the one used in the paper:
    Jointly Learning to Locate and Classify Words using Convolutional Networks by Palaz et. al.
    """

    def __init__(self, out_dim, temp_ratio):
        super(PSC, self).__init__()

        self.temp_ratio = temp_ratio

        self.conv1 = nn.Conv1d(input_dim, 96, 9, 1, 4)
        self.conv2 = nn.Conv1d(96, 96, 11, 1, 5)
        self.conv3 = nn.Conv1d(96, 96, 11, 1, 5)
        self.conv4 = nn.Conv1d(96, 96, 11, 1, 5)
        self.conv5 = nn.Conv1d(96, 96, 11, 1, 5)
        self.conv6 = nn.Conv1d(96, out_dim, 11, 1, 5)
        

    def forward(self, x, x_lengths):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.conv6(x)
        out, frame_scores = self.classifier(x, x_lengths)

        return out, frame_scores

    def classifier(self, x_c, x_lengths):
        """
        This function masks the output of the final conv1d layer and then applies a pooling function.
        The masking is done to black out the paddings applied to the input. This way, gradients are not propagated through the padded regions.
        """

        mask = self.sequence_mask(x_lengths, x_c.size(-1), dtype=torch.uint8).unsqueeze(1).expand(-1, 1000, -1)
        out = x_c.masked_fill(~mask, -np.inf)
        
        # Pooling
        frame_scores = out
        out = (1./self.temp_ratio) * (torch.logsumexp(self.temp_ratio * frame_scores, -1)).squeeze(-1)
 
        
        return out, frame_scores

    def sequence_mask(self, lengths, max_len, device=None, dtype=None):
        assert len(lengths.shape) == 1, "Shape of length should be 1 dimensional."
        max_len = max_len or lengths.max().item()
        # print("MAX_LEN: ", max_len)
        ids = torch.arange(0, max_len, device=lengths.device)
        # print("IDs Shape: ", ids.shape)
        mask = (ids < lengths.unsqueeze(1)).bool()
        # print("MASK SHAPE: ", mask.shape)

        return mask

    



        




