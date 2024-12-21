import torch
import torch.nn as nn

class MPFEM(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.depthwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=0, bias=False, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.fc = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        N, C, H, W = x.size()
            
        dconv1 = self.depthwise_conv(x)
        dconv1 = self.pointwise_conv(dconv1)
        max_out = self.fc(self.max_pool(dconv1))
        avg_out = self.fc(self.avg_pool(dconv1))
        max_score = self.sigmoid(max_out)
        avg_score = self.sigmoid(avg_out)
        
        return max_score * x + avg_score * x + x

class CDEF(nn.Module):
    def __init__(self, in_channels, out_channels, initial_dim):
        super().__init__()
        self.down_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=initial_dim*4, kernel_size=3, stride=1, padding=1)
        self.down_conv2 = nn.Conv2d(in_channels=initial_dim*4, out_channels=initial_dim*4, kernel_size=3, stride=1, padding=1)
        self.down_conv3 = nn.Conv2d(in_channels=initial_dim*4, out_channels=initial_dim*4, kernel_size=3, stride=1, padding=1)
        self.down_conv4 = nn.Conv2d(in_channels=initial_dim*4, out_channels=initial_dim*4, kernel_size=3, stride=1, padding=1)
        
        self.middle_conv = nn.Conv2d(in_channels=initial_dim*4, out_channels=initial_dim*4, kernel_size=3, stride=1, padding=1)
        
        self.up_conv1 = nn.Conv2d(in_channels=initial_dim*4, out_channels=initial_dim*4, kernel_size=3, stride=1, padding=1)
        self.up_conv2 = nn.Conv2d(in_channels=initial_dim*4, out_channels=initial_dim*4, kernel_size=3, stride=1, padding=1)
        self.up_conv3 = nn.Conv2d(in_channels=initial_dim*4, out_channels=initial_dim*4, kernel_size=3, stride=1, padding=1)
        self.up_conv4 = nn.Conv2d(in_channels=initial_dim*4, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        
        self.channel_attention1 = MPFEM(in_channels=initial_dim*4)
        self.channel_attention2 = MPFEM(in_channels=initial_dim*4)
        self.channel_attention3 = MPFEM(in_channels=initial_dim*4)
        self.channel_attention4 = MPFEM(in_channels=initial_dim*4)
        
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        # identity = x    # x: N x 3 x H x W
        sum_tmp = None
        
        x1 = self.down_conv1(x)
        x1 = self.activation(x1)
        sum_tmp = x1
        x2 = self.down_conv2(x1)
        x2 = self.activation(x2)
        x3 = self.down_conv3(x2+sum_tmp)
        sum_tmp += x2
        x3 = self.activation(x3)
        x4 = self.down_conv4(x3+sum_tmp)
        x4 = self.activation(x4)
        
        x5 = self.middle_conv(x4)
        x5 = self.activation(x5)
        
        x6 = self.up_conv1(self.channel_attention4(x5+x4))
        x6 = self.activation(x6)
        sum_tmp = x5
        x7 = self.up_conv2(self.channel_attention3(x6+x3)+sum_tmp)
        x7 = self.activation(x7)
        sum_tmp += x6
        x8 = self.up_conv3(self.channel_attention2(x7+x2)+sum_tmp)
        x8 = self.activation(x8)
        sum_tmp += x7
        x9 = self.up_conv4(self.channel_attention1(x8+x1)+sum_tmp)
        x9 = self.activation(x9)
        
        return x9