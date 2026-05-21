import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """
    Standard Convolutional Neural Network (CNN) for MNIST.
    Architecture: 2 Convolutional layers, Max Pooling, Dropout, and 2 Fully Connected layers.
    """
    def __init__(self):
        super(Net, self).__init__()
        # 1 input channel (grayscale), 32 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # 32 input channels, 64 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # Regularization: 25% dropout and 50% dropout respectively
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        # Fully connected layer from 64 * 12 * 12 (flattened pooled representation)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        # Flatten before passing to fully connected layers
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        # Output log probabilities for NLL_Loss
        output = F.log_softmax(x, dim=1)
        return output
