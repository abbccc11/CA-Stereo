from submodule import *
import math
import gc
import time

class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(
            convbn_3d(in_channels, in_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=False))

        self.conv2 = nn.Sequential(
            convbn_3d(in_channels, in_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=False))

        self.conv3 = nn.Sequential(
            convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=False))

        self.conv4 = nn.Sequential(
            convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=False))

        self.conv5 = nn.Sequential(
            convbn_3d(in_channels * 2, in_channels * 2, 3, 2, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=False))

        self.conv6 = nn.Sequential(
            convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=False))
        
        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels * 2, 4, padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv8 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 4, padding=1 ,stride=2, bias=False),
            nn.BatchNorm3d(in_channels))
        self.feature_att_8 = FeatureAtt(in_channels, 64)
        self.feature_att_16 = FeatureAtt(in_channels *2, 192)
        self.feature_att_32 = FeatureAtt(in_channels * 2, 160)
        self.feature_att_up_16 = FeatureAtt(in_channels * 2, 192)
        self.feature_att_up_8 = FeatureAtt(in_channels, 64)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        conv7 = F.leaky_relu(self.conv7(conv6), negative_slope=0.2, inplace=False)
        conv7 += conv4
        conv8 = F.leaky_relu(self.conv8(conv7), negative_slope=0.2, inplace=False)
        conv8 += conv2
        return conv8
# class hourglass(nn.Module):
#     def __init__(self, in_channels):
#         super(hourglass, self).__init__()
#
#         self.conv1 = nn.Sequential(
#             BasicConv(in_channels, in_channels * 2, is_3d=True, bn=True, relu=True, kernel_size=3,
#                       padding=1, stride=2, dilation=1),
#             BasicConv(in_channels * 2, in_channels * 2, is_3d=True, bn=True, relu=True, kernel_size=3,
#                       padding=1, stride=1, dilation=1))
#
#         self.conv2 = nn.Sequential(
#             BasicConv(in_channels * 2, in_channels * 4, is_3d=True, bn=True, relu=True, kernel_size=3,
#                       padding=1, stride=2, dilation=1),
#             BasicConv(in_channels * 4, in_channels * 4, is_3d=True, bn=True, relu=True, kernel_size=3,
#                       padding=1, stride=1, dilation=1))
#
#         self.conv3 = nn.Sequential(
#             BasicConv(in_channels * 4, in_channels * 6, is_3d=True, bn=True, relu=True, kernel_size=3,
#                       padding=1, stride=2, dilation=1),
#             BasicConv(in_channels * 6, in_channels * 6, is_3d=True, bn=True, relu=True, kernel_size=3,
#                       padding=1, stride=1, dilation=1))
#
#         self.conv3_up = BasicConv(in_channels * 6, in_channels * 4, deconv=True, is_3d=True, bn=True,
#                                   relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
#
#         self.conv2_up = BasicConv(in_channels * 4, in_channels * 2, deconv=True, is_3d=True, bn=True,
#                                   relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
#
#         self.conv1_up = BasicConv(in_channels * 2, 8, deconv=True, is_3d=True, bn=False,
#                                   relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
#
#         self.agg_0 = nn.Sequential(
#             BasicConv(in_channels * 8, in_channels * 4, is_3d=True, kernel_size=1, padding=0, stride=1),
#             BasicConv(in_channels * 4, in_channels * 4, is_3d=True, kernel_size=3, padding=1, stride=1),
#             BasicConv(in_channels * 4, in_channels * 4, is_3d=True, kernel_size=3, padding=1, stride=1), )
#
#         self.agg_1 = nn.Sequential(
#             BasicConv(in_channels * 4, in_channels * 2, is_3d=True, kernel_size=1, padding=0, stride=1),
#             BasicConv(in_channels * 2, in_channels * 2, is_3d=True, kernel_size=3, padding=1, stride=1),
#             BasicConv(in_channels * 2, in_channels * 2, is_3d=True, kernel_size=3, padding=1, stride=1))
#
#         self.feature_att_8 = FeatureAtt(in_channels * 2, 64)
#         self.feature_att_16 = FeatureAtt(in_channels * 4, 192)
#         self.feature_att_32 = FeatureAtt(in_channels * 6, 160)
#         self.feature_att_up_16 = FeatureAtt(in_channels * 4, 192)
#         self.feature_att_up_8 = FeatureAtt(in_channels * 2, 64)
#
#     def forward(self, x, features):
#         conv1 = self.conv1(x)
#         conv1 = self.feature_att_8(conv1, features[1])
#
#         conv2 = self.conv2(conv1)
#         conv2 = self.feature_att_16(conv2, features[2])
#
#         conv3 = self.conv3(conv2)
#         conv3 = self.feature_att_32(conv3, features[3])
#
#         conv3_up = self.conv3_up(conv3)
#         conv2 = torch.cat((conv3_up, conv2), dim=1)
#         conv2 = self.agg_0(conv2)
#         conv2 = self.feature_att_up_16(conv2, features[2])
#
#         conv2_up = self.conv2_up(conv2)
#         conv1 = torch.cat((conv2_up, conv1), dim=1)
#         conv1 = self.agg_1(conv1)
#         conv1 = self.feature_att_up_8(conv1, features[1])
#
#         conv = self.conv1_up(conv1)
#
#         return conv

class EMA(nn.Module):  # 定义一个继承自 nn.Module 的 EMA 类
    def __init__(self, channels, c2=None, factor=16):  # 构造函数，初始化对象
        super(EMA, self).__init__()  # 调用父类的构造函数
        self.groups = factor  # 定义组的数量为 factor，默认值为 32
        assert channels // self.groups > 0  # 确保通道数可以被组数整除
        self.softmax = nn.Softmax(-1)  # 定义 softmax 层，用于最后一个维度
        self.agp = nn.AdaptiveAvgPool3d((1, 1, 1))  # 定义自适应平均池化层，输出大小为 1x1x1
        self.pool_d = nn.AdaptiveAvgPool3d((None, 1, 1))  # 定义自适应平均池化层，只在高度和宽度上池化，保持深度不变
        self.pool_h = nn.AdaptiveAvgPool3d((1, None, 1))  # 定义自适应平均池化层，只在深度和宽度上池化，保持高度不变
        self.pool_w = nn.AdaptiveAvgPool3d((1, 1, None))  # 定义自适应平均池化层，只在深度和高度上池化，保持宽度不变
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)  # 定义组归一化层
        self.conv1x1 = nn.Conv3d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)  # 定义 1x1x1 卷积层
        self.conv3x3 = nn.Conv3d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)  # 定义 3x3x3 卷积层
        self.Sigmoid = nn.Sigmoid()
    def forward(self, x):  # 定义前向传播函数
        b, c, d, h, w = x.size()  # 获取输入张量的大小：批次、通道、深度、高度和宽度
        group_x = x.reshape(b * self.groups, -1, d, h, w)  # 将输入张量重新形状为 (b * 组数, c // 组数, 深度, 高度, 宽度)

        # 在深度上进行池化
        x_d = self.pool_d(group_x)  # 输出形状: (b * groups, c // groups, d, 1, 1)
        # 在高度上进行池化
        x_h = self.pool_h(group_x).permute(0, 1, 3, 2, 4)  # 输出形状: (b * groups, c // groups, 1, h, 1)
        # 在宽度上进行池化
        x_w = self.pool_w(group_x).permute(0, 1, 4, 2, 3)  # 输出形状: (b * groups, c // groups, 1, 1, w)

        # 对池化结果进行拼接，并通过 1x1x1 卷积层
        hw = self.conv1x1(torch.cat([x_d, x_h, x_w], dim=2))  # 拼接后形状: (b * groups, c // groups, d+1+1, h, w)
        # 将卷积结果按深度、高度和宽度分割
        x_d, x_h, x_w = torch.split(hw, [d, h, w], dim=2)  # 分割后各部分形状相同

        # 进行组归一化，并结合深度、高度和宽度的激活结果
        x1 = self.gn(group_x * x_d.sigmoid() * x_h.permute(0, 1, 3, 2, 4).sigmoid() * x_w.permute(0, 1, 4, 2, 3).sigmoid())
        # 通过 3x3x3 卷积层
        x2 = self.conv3x3(group_x)  # 输出形状: (b * groups, c // groups, d, h, w)
        # 对 x1 进行池化、形状变换、并应用 softmax
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))  # 输出形状: (b * groups, 1, 1, 1)
        # 将 x2 重新形状为 (b * 组数, c // 组数, d * h * w)
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # 输出形状: (b * groups, c // groups, d*h*w)
        # 对 x2 进行池化、形状变换、并应用 softmax
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))  # 输出形状: (b * groups, 1, 1, 1)
        # 将 x1 重新形状为 (b * 组数, c // 组数, d * h * w)
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # 输出形状: (b * groups, c // groups, d*h*w)
        # 计算权重
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, d, h, w)  # 输出形状: (b * groups, 1, d, h, w)
        # 应用权重并将形状恢复为原始大小
        return self.Sigmoid(weights.reshape(b, c, d, h, w))


class FeatureFusion(nn.Module):
    def __init__(self, in_channels, reduction=1):
        super(FeatureFusion, self).__init__()
        # 通道注意力模块
        self.att = EMA(in_channels)

    def forward(self, F_up, F_high):
        # 上采样低分辨率特征
        F_up = F.interpolate(F_up, size=F_high.shape[2:], mode='trilinear')
        F_concat = F_up + F_high
        # 对每个特征分别计算通道注意力
        v = self.att(F_concat)
        v1 = 1.0 - v
        x3 = F_up * v
        x4 = F_high * v1
        x = x3 + x4
        return x




