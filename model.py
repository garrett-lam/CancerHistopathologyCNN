import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    '''
    This basically makes the conv -> bn -> relu -> conv -> bn -> relu -> pool block structure you have (You made 5 of them)
    in_channels: input for first conv layer in the block
    out_channels: output for first conv layer, and input and output for second conv layer in the block. also used for nn.BatchNorm2d
    '''
    def __init__(self, relu_type, pool_type, in_channels, out_channels, elu_val = 1, lrelu_val = .01):
        super(ConvBlock, self).__init__()

        if relu_type == 'relu':
            self.act = nn.ReLU()
        elif relu_type == 'lrelu':
            self.act = nn.LeakyReLU(lrelu_val)
        else:
            self.act = nn.ELU(elu_val)

        if pool_type == 'avg':
            self.pool = nn.AvgPool2d(2, 2)
        else:
            self.pool = nn.MaxPool2d(2, 2, ceil_mode = True)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
    
    def forward(self, x):
        return self.pool(self.act(self.bn(self.conv2(self.act(self.bn(self.conv1(x)))))))
        


class Model(nn.Module):
    def __init__(self, relu_type, pool_type, elu_val = 1, lrelu_val = .01): # elu_val refers to the alpha arg, lrelu_val refers to the negative slope arg
        super(Model, self).__init__()
        self.layers = nn.Sequential(
        ConvBlock(relu_type, pool_type, 3, 16, elu_val, lrelu_val),
        ConvBlock(relu_type, pool_type, 16, 32, elu_val, lrelu_val),
        ConvBlock(relu_type, pool_type, 32, 64, elu_val, lrelu_val),
        ConvBlock(relu_type, pool_type, 64, 128, elu_val, lrelu_val),
        ConvBlock(relu_type, pool_type, 128, 256, elu_val, lrelu_val),
        nn.Flatten(),
        nn.Linear(256 * 24 * 24, 1024),
        nn.BatchNorm1d(1024),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),

        nn.Linear(512, 10) # i'm not sure why output is 10 when there are 5 classes, but I'll leave it as it is. feel free to change it 
        )
    
    def forward(self, x):
        return self.layers(x)

possible_activation_inputs = ['relu', 'lrelu', 'anything else'] # just for reference (anything else will end up using ELU)
possible_pool_inputs = ['avg' , 'anything else'] # just for reference (anything else will end up using maxpooling)
model = Model('relu', 'max') # example instance where activation is ReLU and pooling is maxpooling