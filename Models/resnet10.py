import torch
import torch.nn as nn

class ResBlock(nn.Module):
    
    def __init__(self, input_size, output_size, stride = 1, padding = 1):
        super(ResBlock, self).__init__()
        self.convA = nn.Conv2d(input_size, output_size, 3, stride = stride, padding = 1)
        self.convB = nn.Conv2d(output_size, output_size, 3, stride = 1, padding = 1)
        self.downsampling = nn.Conv2d(input_size, output_size, 3, stride = stride, padding = 1) if stride != 1 else None
        self.relu = nn.ReLU()
        self.output_size = output_size
        self.input_size = input_size
        self.stride = stride
        
    def forward(self, x):
        if self.output_size != self.input_size and self.downsampling is None:
            res = x.repeat(1,2,1,1)
        else:
            res = x
        if self.downsampling is not None:
            res = self.downsampling(x)
        output = self.relu(self.convB(self.relu((self.convA(x)))).add(res))
        return output
    
class ResNet(nn.Module):
    
    def __init__(self, img_size):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, (7, 7), stride = 2)
        self.maxp = nn.MaxPool2d(kernel_size = 3, stride = 2)
        self.conv2_x = ResBlock(64, 64, stride = 1)
        self.conv3_x = ResBlock(64, 128)
        self.conv4_x = ResBlock(128, 256)
        self.conv5_x = ResBlock(256, 512, stride = 2)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(2048, 100)
        self.avgp = nn.AvgPool2d(kernel_size = 3, stride = 2)
        
    def forward(self, x):
        layer1 = self.maxp(self.conv1(x))
        resblock1 = self.conv2_x(layer1)
        resblock2 = self.conv3_x(resblock1)
        resblock3 = self.conv4_x(resblock2)
        resblock4 = self.conv5_x(resblock3)
        y = self.fc1(self.flatten(self.avgp(resblock4)))
        return y