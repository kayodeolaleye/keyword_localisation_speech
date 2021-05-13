import torch.nn as nn
import torch
import torch.nn.functional as F

class CNNPool(nn.Module):
    """
    This class defines the ConvNet architecture for the text-speech detection models. 
    It subclasses the nn.Module class which is the base class for all PyTorch neural network modules.
    """

    def __init__(self, out_dim):
        """
        Initialises some of the parameters of the model.
        """
        super(CNNPool, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(39, 64, 9, 1, 4),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(64, 256, 11, 1, 5),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(256, 1024, 11, 1, 5),
            nn.ReLU()
        )
        self.classify = nn.Sequential(
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, out_dim),
            nn.Dropout(p=0.0)
        )

    def forward(self, x):
        """
        This function applies a sequence of functions to an input of size (N, in_dim, L)
        N: number of batch
        in_dim: the dimension of the input signal
        L: the length of each input signal (thesame for each signal because of padding)

        Args:
        x (Tensor): a Tensor input of size (N, in_dim, L)

        Output:
        cnn (Tensor): a Tensor of size (N, options_dict["output_dim]) 
        """
        x = self.features(x)
        x = torch.max(x, dim=2).values
        x = self.classify(x)
        return x
