import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_bn_act(in_channel,out_channel, kernel_size, strides, padding, dilation_rate):
    """
    定义卷积 -> 批归一化 -> 激活函数的模块
    """
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channel,
                  out_channels=out_channel,
                  kernel_size=kernel_size,
                  stride=strides,
                  padding=padding,
                  dilation=dilation_rate,
                  bias=False),
        nn.BatchNorm2d(out_channel),
        nn.LeakyReLU()
    )

class Refinement(nn.Module):
    def __init__(self):
        super(Refinement, self).__init__()
        self.conv1 = conv_bn_act(4, 32, kernel_size=3, strides=1, padding=1, dilation_rate=1)
        self.conv2 = conv_bn_act(32, 32, kernel_size=3, strides=1, padding=1, dilation_rate=1)
        self.conv3 = conv_bn_act(32, 32, kernel_size=3, strides=1, padding=2, dilation_rate=2)
        self.conv4 = conv_bn_act(32, 32, kernel_size=3, strides=1, padding=3, dilation_rate=3)
        self.conv5 = conv_bn_act(32, 32, kernel_size=3, strides=1, padding=1, dilation_rate=1)
        self.conv6 = nn.Conv2d(in_channels=32,  out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward( self, disparity, rgb, gx, gy):
        disparity = disparity.unsqueeze(1)
       
        # 拼接所有输入
        concat = torch.cat([disparity, rgb, gx, gy], dim=1)

        # 残差学习
        delta1 = self.conv1(concat)
        delta2 = self.conv2(delta1)
        delta3 = self.conv3(delta2 + delta1)  # 跨层连接 delta2 + delta1
        delta4 = self.conv4(delta3 + delta1)  # 跨层连接 delta3 + delta1
        delta5 = self.conv5(delta4)

        # 生成最终的视差图
        delta = self.conv6(delta5)
        disp_final = disparity + delta

        return disp_final