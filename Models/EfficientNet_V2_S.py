import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights


class EfficientNet_V2_S(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNet_V2_S, self).__init__()

        # Import EfficientNet_V2_S weights
        self.model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)

        # Customize to our dataset 
        in_features = self.model.classifier[1].in_features  # Get the number of input features to the Linear layer
        self.model.classifier[1] = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)