import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    '''
    This basically makes the conv -> bn -> relu -> conv -> bn -> relu -> pool block structure you have (You made 5 of them)
    in_channels: input for first conv layer in the block
    out_channels: output for first conv layer, and input and output for second conv layer in the block. also used for nn.BatchNorm2d
    '''
    def __init__(self, relu_type, pool_type, in_channels, out_channels, elu_val = 1, lrelu_val = 0.01):
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
            self.pool = nn.MaxPool2d(2, 2)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.dropout = nn.Dropout2d(0.5)
    
    def forward(self, x):
        return self.pool(self.act(self.bn(self.conv2(self.dropout(self.act(self.bn(self.conv1(x))))))))
        


class BaseCNN(nn.Module):
    def __init__(self, relu_type, pool_type, img_size, elu_val = 1, lrelu_val = 0.01): # elu_val refers to the alpha arg, lrelu_val refers to the negative slope arg
        super(BaseCNN, self).__init__()
        # self.layers = nn.Sequential(
        #     ConvBlock(relu_type, pool_type, 3, 16, elu_val, lrelu_val),
        #     ConvBlock(relu_type, pool_type, 16, 32, elu_val, lrelu_val),
        #     ConvBlock(relu_type, pool_type, 32, 64, elu_val, lrelu_val),
        #     ConvBlock(relu_type, pool_type, 64, 128, elu_val, lrelu_val),
        #     ConvBlock(relu_type, pool_type, 128, 256, elu_val, lrelu_val),
        #     nn.Flatten(),
        #     nn.Linear(256 * int((img_size / 32)**2), 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.Linear(1024, 512),
        #     nn.BatchNorm1d(512),
        #     nn.Linear(512, 5)
        # )

        if relu_type == 'relu':
            self.act = nn.ReLU()
        elif relu_type == 'lrelu':
            self.act = nn.LeakyReLU(lrelu_val)
        else:
            self.act = nn.ELU(elu_val)
        
        if pool_type == 'avg':
            self.pool = nn.AvgPool2d(2, 2)
        else:
            self.pool = nn.MaxPool2d(2, 2)

        self.layers = nn.Sequential(
            # Block1 (different to the other ConvBlocks)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            self.pool,
            # Block 2 and 3
            ConvBlock(relu_type, pool_type, 64, 128, elu_val, lrelu_val),
            ConvBlock(relu_type, pool_type, 128, 256, elu_val, lrelu_val),
            # Feed-forward linear layers
            nn.Flatten(),
            nn.Linear(256 * int((img_size / 8)**2), 1024),
            nn.BatchNorm1d(1024),
            self.act,
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            self.act,
            nn.Linear(512, 5)
        )

    
    def forward(self, x):
        return self.layers(x)
