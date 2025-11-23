from torch import nn
from torch.nn import functional as F

class Adapter(nn.Module):
    def __init__(self, imput_channels, output_channels, out_size):
        super().__init__()
        self.conv = nn.Conv2d(imput_channels, output_channels, kernel_size=1)
        self.out_size = out_size
        
    def forward(self, feat):
        feat = self.conv(feat)
        feat = F.interpolate(feat, size=self.out_size, mode='bilinear', align_corners=False)
        return feat
    
