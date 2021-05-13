"""
Code borrowed from https://github.com/utkuozbulak/pytorch-cnn-visualizations
"""

from operator import mod
import numpy as np
import models
import torch
import torch.nn.functional as F

class CamExtractor():
    """ Extracts CAM features from the model """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at a given layer
        """
        conv_output = None
        for module_pos, module in enumerate(self.model.features.children(), 1):
            x = module(x) # Forward
            if int(module_pos) == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
        Does a full forward pass on the model
        """
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = torch.max(F.relu(x), dim=2).values
        x = self.model.classify(x)
        return conv_output, x

class GradCAM():
    """
    Produces class activation map
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, x, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(x)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())

        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
    
        # Zero grads
        self.model.features.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output.cuda(), retain_graph=True) # there is a bug here
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.cpu().numpy()[0]
        # Get convolution outputs
        target = conv_output.data.cpu().numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=1) # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        return cam


