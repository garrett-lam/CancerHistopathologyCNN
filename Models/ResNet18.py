import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18(nn.Module):
    def __init__(self, num_classes, img_size):
        super(ResNet18, self).__init__()

        # Import ResNet18 weights
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Customize to our dataset 
        in_features = self.model.fc.in_features # Get the number of input features to the Linear layer
        self.model.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)