import torch.nn as nn
from torchvision.models import vgg11_bn, VGG11_BN_Weights


class VGG11_bn(nn.Module):
    def __init__(self, num_classes, img_size):
        super(VGG11_bn, self).__init__()

        # Import VGG11_BN weights
        self.model = vgg11_bn(weights=VGG11_BN_Weights.DEFAULT)

        # Customize to our dataset 
        in_features = self.model.classifier[6].in_features # Get the number of input features to the Linear layer
        self.model.classifier[6] = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)