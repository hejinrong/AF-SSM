# import torch
# from mamba_ssm import Mamba
#
# batch, length, dim = 2, 64, 16 #(B,L,F)
# x = torch.randn(batch, length, dim).to("cuda")
# model = Mamba(
#     # This module uses roughly 3 * expand * d_model^2 parameters
#     d_model=dim, # Model dimension d_model
#     d_state=16,  # SSM state expansion factor
#     d_conv=4,    # Local convolution width
#     expand=2,    # Block expansion factor
# ).to("cuda")
# print(model)
# print("++++++++++++++++++++++++++++++")
# y = model(x)
# print(x.shape,y.shape)
# # assert y.shape == x.shape
import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange

# 假设这些是您的四个特征图
tensor1 = torch.randn(1, 96, 128, 128)  # 最底层的特征图
tensor2 = torch.randn(1, 192, 64, 64)
tensor3 = torch.randn(1, 384, 32, 32)
tensor4 = torch.randn(1, 768, 16, 16)  # 最顶层的特征图


class Tokenizer01(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):  # 卷积然后打成patch
        return rearrange(x, 'b c h w -> b (h w) c')


class Tokenizer02(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):  # 卷积然后打成patch
        return rearrange(x, 'b (h w) c -> b c 16 16')


# 定义一个简单的卷积块
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


input = torch.randn(64, 768, 16, 16)
to01 = Tokenizer01()
to02 = Tokenizer02()
out01 = to01(input)
out02 = to02(out01)
print(out01.shape)
