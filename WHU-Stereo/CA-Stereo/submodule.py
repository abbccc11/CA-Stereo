import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()

        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = nn.LeakyReLU()(x)  # , inplace=True)
        return x


class Conv2x(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, keep_concat=True, bn=True,
                 relu=True, keep_dispc=False):
        super(Conv2x, self).__init__()
        self.concat = concat
        self.is_3d = is_3d
        if deconv and is_3d:
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel,
                                   stride=stride, padding=padding)
        else:
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel,
                                   stride=2, padding=1)

        if self.concat:
            mul = 2 if keep_concat else 1
            self.conv2 = BasicConv(out_channels * 2, out_channels * mul, False, is_3d, bn, relu, kernel_size=3,
                                   stride=1, padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1,
                                   padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(
                x,
                size=(rem.shape[-2], rem.shape[-1]),
                mode='nearest')
        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem
        x = self.conv2(x)
        return x


class BasicConv_IN(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, IN=True, relu=True, **kwargs):
        super(BasicConv_IN, self).__init__()

        self.relu = relu
        self.use_in = IN
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_in:
            x = self.IN(x)
        if self.relu:
            x = nn.LeakyReLU()(x)  # , inplace=True)
        return x


class Conv2x_IN(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, keep_concat=True, IN=True,
                 relu=True, keep_dispc=False):
        super(Conv2x_IN, self).__init__()
        self.concat = concat
        self.is_3d = is_3d
        if deconv and is_3d:
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv_IN(in_channels, out_channels, deconv, is_3d, IN=True, relu=True, kernel_size=kernel,
                                      stride=stride, padding=padding)
        else:
            self.conv1 = BasicConv_IN(in_channels, out_channels, deconv, is_3d, IN=True, relu=True, kernel_size=kernel,
                                      stride=2, padding=1)

        if self.concat:
            mul = 2 if keep_concat else 1
            self.conv2 = BasicConv_IN(out_channels * 2, out_channels * mul, False, is_3d, IN, relu, kernel_size=3,
                                      stride=1, padding=1)
        else:
            self.conv2 = BasicConv_IN(out_channels, out_channels, False, is_3d, IN, relu, kernel_size=3, stride=1,
                                      padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(
                x,
                size=(rem.shape[-2], rem.shape[-1]),
                mode='nearest')
        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem
        x = self.conv2(x)
        return x


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


# def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
#     B, C, H, W = refimg_fea.shape
#     volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
#     for i in range(maxdisp):
#         if i > 0:
#             volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
#                                                            num_groups)
#         else:
#             volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
#     volume = volume.contiguous()
#     return volume
#
def build_gwc_volume(refimg_fea, targetimg_fea, mindisp, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp - mindisp, H, W])
    for i in range(mindisp, maxdisp):
        index = i - mindisp
        if i > 0:
            # Shift refimg_fea to the right and targetimg_fea to the left
            volume[:, :, index, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                               num_groups)
        elif i < 0:
            # Shift refimg_fea to the left and targetimg_fea to the right
            volume[:, :, index, :, :i] = groupwise_correlation(refimg_fea[:, :, :, :i], targetimg_fea[:, :, :, -i:],
                                                               num_groups)
        else:
            # No shift
            volume[:, :, index, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume


def norm_correlation(fea1, fea2):
    cost = torch.mean(
        ((fea1 / (torch.norm(fea1, 2, 1, True) + 1e-05)) * (fea2 / (torch.norm(fea2, 2, 1, True) + 1e-05))), dim=1,
        keepdim=True)
    return cost


def build_norm_correlation_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 1, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = norm_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i])
        else:
            volume[:, :, i, :, :] = norm_correlation(refimg_fea, targetimg_fea)
    volume = volume.contiguous()
    return volume


def correlation(fea1, fea2):
    cost = torch.sum((fea1 * fea2), dim=1, keepdim=True)
    return cost


def build_correlation_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 1, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i])
        else:
            volume[:, :, i, :, :] = correlation(refimg_fea, targetimg_fea)
    volume = volume.contiguous()
    return volume


#
# def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
#     B, C, H, W = refimg_fea.shape
#     volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
#     for i in range(maxdisp):
#         if i > 0:
#             volume[:, :C, i, :, :] = refimg_fea[:, :, :, :]
#             volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
#         else:
#             volume[:, :C, i, :, :] = refimg_fea
#             volume[:, C:, i, :, :] = targetimg_fea
#     volume = volume.contiguous()
#     return volume
def build_concat_volume(refimg_fea, targetimg_fea, mindisp, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp - mindisp, H, W])
    for i in range(mindisp, maxdisp):
        index = i - mindisp
        if i > 0:
            volume[:, :C, index, :, :] = refimg_fea
            volume[:, C:, index, :, i:] = targetimg_fea[:, :, :, :-i]
        elif i < 0:
            volume[:, :C, index, :, :i] = refimg_fea[:, :, :, -i:]
            volume[:, C:, index, :, :] = targetimg_fea
        else:
            volume[:, :C, index, :, :] = refimg_fea
            volume[:, C:, index, :, :] = targetimg_fea
    volume = volume.contiguous()
    return volume


# def disparity_regression(x, maxdisp):
#     assert len(x.shape) == 4
#     disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
#     disp_values = disp_values.view(1, maxdisp, 1, 1)
#     return torch.sum(x * disp_values, 1, keepdim=True)
def disparity_regression(x, mindisp, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(mindisp, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp - mindisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=True)


class FeatureAtt(nn.Module):
    def __init__(self, cv_chan, feat_chan):
        super(FeatureAtt, self).__init__()

        self.feat_att = nn.Sequential(
            BasicConv(feat_chan, feat_chan // 2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(feat_chan // 2, cv_chan, 1))

    def forward(self, cv, feat):
        '''
        '''
        feat_att = self.feat_att(feat).unsqueeze(2)
        cv = torch.sigmoid(feat_att) * cv
        return cv


def context_upsample(disp_low, up_weights):
    ###
    # cv (b,1,h,w)
    # sp (b,9,4*h,4*w)
    ###
    b, c, h, w = disp_low.shape

    disp_unfold = F.unfold(disp_low.reshape(b, c, h, w), 3, 1, 1).reshape(b, -1, h, w)
    disp_unfold = F.interpolate(disp_unfold, (h * 4, w * 4), mode='nearest').reshape(b, 9, h * 4, w * 4)

    disp = (disp_unfold * up_weights).sum(1)

    return disp
class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()

        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = nn.LeakyReLU()(x)  # , inplace=True)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(SoftPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x):
        x_exp = torch.exp(x)
        sum_pool = F.avg_pool2d(x_exp, self.kernel_size, self.stride, self.padding)
        soft_pool = sum_pool / F.avg_pool2d(torch.ones_like(x), self.kernel_size, self.stride, self.padding)
        return soft_pool

class Conv2x(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, keep_concat=True, bn=True,
                 relu=True, keep_dispc=False):
        super(Conv2x, self).__init__()
        self.concat = concat
        self.is_3d = is_3d
        if deconv and is_3d:
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel,
                                   stride=stride, padding=padding)
        else:
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel,
                                   stride=2, padding=1)

        if self.concat:
            mul = 2 if keep_concat else 1
            self.conv2 = BasicConv(out_channels * 2, out_channels * mul, False, is_3d, bn, relu, kernel_size=3,
                                   stride=1, padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1,
                                   padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(
                x,
                size=(rem.shape[-2], rem.shape[-1]),
                mode='nearest')
        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem
        x = self.conv2(x)
        return x


def convbn(in_channels, out_channels, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_channels))


def convbn_group(in_channels, out_channels, groups, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, groups=groups, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_channels))


def convbn_3d(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=pad, bias=False),
                         nn.BatchNorm3d(out_channels))


def convbn_3d_group(in_channels, out_channels, groups, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, groups=groups, kernel_size=kernel_size, stride=stride,
                                   padding=pad, bias=False),
                         nn.BatchNorm3d(out_channels))


def convgn(in_channels, out_channels, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.GroupNorm(4, out_channels))


def convgn_group(in_channels, out_channels, groups, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, groups=groups, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.GroupNorm(4, out_channels))


def convgn_3d(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=pad, bias=False),
                         nn.GroupNorm(4, out_channels))


def convgn_3d_group(in_channels, out_channels, groups, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, groups=groups, kernel_size=kernel_size, stride=stride,
                                   padding=pad, bias=False),
                         nn.GroupNorm(4, out_channels))


def convbn_3d_1kk(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=(1, kernel_size, kernel_size), stride=(1, stride, stride),
                  padding=(0, pad, pad), bias=False),
        nn.BatchNorm3d(out_channels))


def convbn_3d_new(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=(kernel_size, 1, 1), stride=(stride, 1, 1),
                                   padding=(pad, 0, 0), bias=False),
                         nn.Conv3d(out_channels, out_channels, kernel_size=(1, kernel_size, 1), stride=(1, stride, 1),
                                   padding=(0, pad, 0), bias=False),
                         nn.Conv3d(out_channels, out_channels, kernel_size=(1, 1, kernel_size), stride=(1, 1, stride),
                                   padding=(0, 0, pad), bias=False),
                         nn.BatchNorm3d(out_channels))


def conv_3d_new(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=(kernel_size, 1, 1), stride=(stride, 1, 1),
                                   padding=(pad, 0, 0), bias=False),
                         nn.Conv3d(out_channels, out_channels, kernel_size=(1, kernel_size, 1), stride=(1, stride, 1),
                                   padding=(0, pad, 0), bias=False),
                         nn.Conv3d(out_channels, out_channels, kernel_size=(1, 1, kernel_size), stride=(1, 1, stride),
                                   padding=(0, 0, pad), bias=False))


def convTrans_3d_new(in_channels, out_channels, kernel_size, pad, output_pad, stride):
    return nn.Sequential(
        nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(kernel_size, 1, 1), stride=(stride, 1, 1),
                           padding=(pad, 0, 0), output_padding=(output_pad, 0, 0), bias=False),
        nn.ConvTranspose3d(out_channels, out_channels, kernel_size=(1, kernel_size, 1), stride=(1, stride, 1),
                           padding=(0, pad, 0), output_padding=(0, output_pad, 0), bias=False),
        nn.ConvTranspose3d(out_channels, out_channels, kernel_size=(1, 1, kernel_size), stride=(1, 1, stride),
                           padding=(0, 0, pad), output_padding=(0, 0, output_pad), bias=False))


def convbn_3d_dw(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                   padding=pad, bias=False, groups=in_channels),
                         nn.Conv3d(in_channels, out_channels, kernel_size=1),
                         nn.BatchNorm3d(out_channels))


def conv_3d_dw(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                   padding=pad, bias=False, groups=in_channels),
                         nn.Conv3d(in_channels, out_channels, kernel_size=1))


def convTrans_3d_dw(in_channels, out_channels, kernel_size, pad, output_pad, stride):
    return nn.Sequential(nn.ConvTranspose3d(in_channels, out_channels, kernel_size=1),
                         nn.ConvTranspose3d(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                            padding=pad, output_padding=output_pad, bias=False, groups=out_channels))


def disparity_regression(x, mindisp, maxdisp):
    assert len(x.shape) == 4
    B, D, H, W = x.shape  # D should be maxdisp - mindisp
    disp_values = torch.arange(mindisp, maxdisp, dtype=x.dtype, device=x.device).view(1, D, 1, 1)
    return torch.sum(x * disp_values, 1)


def build_concat_volume(refimg_fea, targetimg_fea,mindisp, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp - mindisp, H, W])
    for i in range(mindisp, maxdisp):
        index = i - mindisp
        if i > 0:
            volume[:, :C, index, :, :] = refimg_fea
            volume[:, C:, index, :, i:] = targetimg_fea[:, :, :, :-i]
        elif i < 0:
            volume[:, :C, index, :, :i] = refimg_fea[:, :, :, -i:]
            volume[:, C:, index, :, :] = targetimg_fea
        else:
            volume[:, :C, index, :, :] = refimg_fea
            volume[:, C:, index, :, :] = targetimg_fea
    volume = volume.contiguous()
    return volume


def build_gwc_volume_cos(refimg_fea, targetimg_fea, maxdisp, num_groups):
    refimg_fea = refimg_fea / (torch.sum(refimg_fea ** 2, dim=1, keepdim=True).pow(1 / 2) + 1e-05)
    targetimg_fea = targetimg_fea / (torch.sum(targetimg_fea ** 2, dim=1, keepdim=True).pow(1 / 2) + 1e-05)
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume

def group_correlation(left_feature, right_feature, groups=8):
    B, C, D, H, W = left_feature.size()
    G = groups
    assert C % G == 0 
    C_per_group = C // G

    # 重塑特征图为 (B, G, C_per_group, D, H, W)
    left_group = left_feature.view(B, G, C_per_group, D, H, W)
    right_group = right_feature.view(B, G, C_per_group, D, H, W)

    # 计算点积相似度
    correlation = torch.sum(left_group * right_group, dim=2)  # (B, G, D, H, W)

    return correlation

def groupwise_correlation_norm(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    fea1 = fea1.view([B, num_groups, channels_per_group, H, W])
    fea2 = fea2.view([B, num_groups, channels_per_group, H, W])
    cost = ((fea1 / (torch.norm(fea1, 2, 2, True) + 1e-05)) * (fea2 / (torch.norm(fea2, 2, 2, True) + 1e-05))).mean(
        dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def regression_topk(cost, disparity_samples, k):

    _, ind = cost.sort(1, True)
    pool_ind = ind[:, :k]
    cost = torch.gather(cost, 1, pool_ind)
    prob = F.softmax(cost, 1)
    disparity_samples = torch.gather(disparity_samples, 1, pool_ind)    
    pred = torch.sum(disparity_samples * prob, dim=1, keepdim=True)
    return pred

def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost

def build_gwc_volume(refimg_fea, targetimg_fea, mindisp, maxdisp, num_groups):
    
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp - mindisp, H, W])
    for i in range(mindisp, maxdisp):
        index = i - mindisp
        if i > 0:
            # Shift refimg_fea to the right and targetimg_fea to the left
            volume[:, :, index, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i], num_groups)
        elif i < 0:
            # Shift refimg_fea to the left and targetimg_fea to the right
            volume[:, :, index, :, :i] = groupwise_correlation(refimg_fea[:, :, :, :i], targetimg_fea[:, :, :, -i:], num_groups)
        else:
            # No shift
            volume[:, :, index, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume



def build_gwc_volume_norm(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation_norm(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                                num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation_norm(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume


def patch_aggregation(gwc_volume, patch_weight):
    gwc_volume_pad = torch.nn.functional.pad(gwc_volume, pad=(1, 1, 1, 1), mode="constant", value=0)
    gwc_volume_pad_unfold = gwc_volume_pad.unfold(3, 3, 1).unfold(4, 3, 1)  # [N,C,D,H,W,3,3]
    gwc_volume_pad_unfold = gwc_volume_pad_unfold.contiguous().view(gwc_volume.shape[0], gwc_volume.shape[1],
                                                                    gwc_volume.shape[2], gwc_volume.shape[3],
                                                                    gwc_volume.shape[4], -1).permute(0, 1, 5, 2, 3, 4)
    gwc_volume_pad_unfold = patch_weight.view(gwc_volume.shape[0], gwc_volume.shape[1], 1, gwc_volume.shape[2],
                                              gwc_volume.shape[3], gwc_volume.shape[4]) * gwc_volume_pad_unfold
    gwc_volume = torch.sum(gwc_volume_pad_unfold, dim=2)
    return gwc_volume


class Build_gwc_volume_unfold(nn.Module):
    def __init__(self, maxdisp):
        self.maxdisp = maxdisp
        super(Build_gwc_volume_unfold, self).__init__()
        self.unfold = nn.Unfold((1, maxdisp), 1, 0, 1)
        self.left_pad = nn.ZeroPad2d((maxdisp - 1, 0, 0, 0))

    def forward(self, refimg_fea, targetimg_fea, num_groups):
        B, C, H, W = refimg_fea.shape
        unfolded_targetimg_fea = self.unfold(self.left_pad(targetimg_fea)).reshape(
            B, num_groups, C // num_groups, self.maxdisp, H, W)
        refimg_fea = refimg_fea.view(B, num_groups, C // num_groups, 1, H, W)
        volume = (refimg_fea * unfolded_targetimg_fea).sum(2)
        volume = torch.flip(volume, [2])
        return volume


def build_gwc_volume_v1(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, (2 * i):] = groupwise_correlation(refimg_fea[:, :, :, (2 * i):],
                                                                 targetimg_fea[:, :, :, :-(2 * i)],
                                                                 num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume


def build_correlation_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_ones([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class BasicBlock_gn(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock_gn, self).__init__()

        self.conv1 = nn.Sequential(convgn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convgn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class BasicBlock_groups(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, groups, stride, downsample, pad, dilation):
        super(BasicBlock_groups, self).__init__()

        self.conv1 = nn.Sequential(convbn_group(inplanes, planes, groups, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_group(planes, planes, groups, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class attention_block(nn.Module):
    def __init__(self, channels_3d, num_heads=8, block=4):
        """
        ws 1 for stand attention
        """
        super(attention_block, self).__init__()
        self.block = block
        self.dim_3d = channels_3d
        self.num_heads = num_heads
        head_dim_3d = self.dim_3d // num_heads
        self.scale_3d = head_dim_3d ** -0.5
        self.qkv_3d = nn.Linear(self.dim_3d, self.dim_3d * 3, bias=True)
        self.final1x1 = torch.nn.Conv3d(self.dim_3d, self.dim_3d, 1)


    def forward(self, x):

        B, C, D0, H0, W0 = x.shape
        pad_f = pad_l = pad_t = 0
        pad_d = (self.block[0] - D0 % self.block[0]) % self.block[0]
        pad_r = (self.block[2] - W0 % self.block[2]) % self.block[2]
        pad_b = (self.block[1] - H0 % self.block[1]) % self.block[1]
        x = F.pad(x, ( pad_l, pad_r, pad_t, pad_b,pad_f,pad_d))
        B, C, D, H, W = x.shape
        d, h, w = D//self.block[0], H//self.block[1], W//self.block[2]
        
        x = x.view(B, C, d,self.block[0], h, self.block[1], w, self.block[2]).permute(0, 2, 4, 6, 3, 5, 7, 1)

        qkv_3d = self.qkv_3d(x).reshape(B, d*h*w, self.block[0]*self.block[1]*self.block[2], 3, self.num_heads,
                                            C // self.num_heads).permute(3, 0, 1, 4, 2, 5)  #[3,B,d*h*w,num_heads,blocks,C//num_heads]
        q_3d, k_3d, v_3d = qkv_3d[0], qkv_3d[1], qkv_3d[2]
        attn = (q_3d @ k_3d.transpose(-2, -1)) * self.scale_3d
        if pad_d > 0 or pad_r > 0 or pad_b > 0:
            mask = torch.zeros((1, D, H, W), device=x.device)
            mask[:,-pad_d:, :, :].fill_(1)
            mask[:,:, -pad_b:, :].fill_(1)
            mask[:,:, :, -pad_r:].fill_(1)
            mask = mask.reshape(1, d,self.block[0], h, self.block[1], w, self.block[2]).permute(0, 1, 3, 5, 2, 4, 6).reshape(1,  d*h*w, self.block[0]*self.block[1]*self.block[2])
            attn_mask = mask.unsqueeze(2) - mask.unsqueeze(3)  # 1, _h*_w, self.block*self.block, self.block*self.block
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-1000.0)).masked_fill(attn_mask == 0, float(0.0))
            
            attn = attn + attn_mask.unsqueeze(2).repeat(B, 1,  self.num_heads, 1, 1)

        attn = torch.softmax(attn, dim=-1)

        x = (attn @ v_3d).view(B, d, h ,w, self.num_heads, self.block[0], self.block[1], self.block[2], -1).permute(0,4,8,1,5,2,6,3,7)
        x = x.reshape(B, C, D, H, W)
        if pad_d > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :, :D0, :H0, :W0]
        return self.final1x1(x)

def disparity_variance(x, mindisp,maxdisp, disparity):
    # the shape of disparity should be B,1,H,W, return is the variance of the cost volume [B,1,H,W]
    assert len(x.shape) == 4
    disp_values = torch.arange(mindisp, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp-mindisp, 1, 1)
    disp_values = (disp_values - disparity) ** 2
    return torch.sum(x * disp_values, 1, keepdim=True)

def disparity_variance_confidence(x, disparity_samples, disparity):
    # the shape of disparity should be B,1,H,W, return is the uncertainty estimation
    assert len(x.shape) == 4
    disp_values = (disparity - disparity_samples) ** 2
    return torch.sum(x * disp_values, 1, keepdim=True)


class SpatialTransformer(nn.Module):
    def __init__(self):
        super(SpatialTransformer, self).__init__()

    def forward(self, left_input, right_input, disparity_samples):
        """
        Disparity Sample Cost Evaluator
        Description:
                Given the left image features, right iamge features and the disparity samples, generates:
                    - Warped right image features

        Args:
            :left_input: Left Image Features
            :right_input: Right Image Features
            :disparity_samples:  Disparity Samples

        Returns:
            :warped_right_feature_map: right iamge features warped according to input disparity.
            :left_feature_map: expanded left image features.
        """

        device = left_input.get_device()
        left_y_coordinate = torch.arange(0.0, left_input.size()[3], device=device).repeat(left_input.size()[2])
        # left_y_coordinate = torch.arange(0.0, left_input.size()[3]).repeat(left_input.size()[2])
        left_y_coordinate = left_y_coordinate.view(left_input.size()[2], left_input.size()[3])
        # left_y_coordinate = torch.clamp(left_y_coordinate, min=0, max=left_input.size()[3] - 1)
        left_y_coordinate = left_y_coordinate.expand(left_input.size()[0], -1, -1)

        right_feature_map = right_input.expand(disparity_samples.size()[1], -1, -1, -1, -1).permute([1, 2, 0, 3, 4])
        left_feature_map = left_input.expand(disparity_samples.size()[1], -1, -1, -1, -1).permute([1, 2, 0, 3, 4])

        disparity_samples = disparity_samples.float()

        right_y_coordinate = left_y_coordinate.expand(
            disparity_samples.size()[1], -1, -1, -1).permute([1, 0, 2, 3]) - disparity_samples

        right_y_coordinate_1 = right_y_coordinate
        right_y_coordinate = torch.clamp(right_y_coordinate, min=0, max=(right_input.size()[3] - 1))

        warped_right_feature_map = torch.gather(right_feature_map, dim=4,
                                                index=right_y_coordinate.expand(right_input.size()[1], -1, -1, -1,
                                                                                -1).permute([1, 0, 2, 3, 4]).long())

        right_y_coordinate_1 = right_y_coordinate_1.unsqueeze(1)
        warped_right_feature_map = (1 - ((right_y_coordinate_1 < 0) + \
                                         (right_y_coordinate_1 > right_input.size()[3] - 1)).float()) * \
                                   (warped_right_feature_map) + torch.zeros_like(warped_right_feature_map)

        return warped_right_feature_map, left_feature_map


def SpatialTransformer_grid(x, y, disp_range_samples):
    bs, channels, height, width = y.size()
    ndisp = disp_range_samples.size()[1]

    mh, mw = torch.meshgrid([torch.arange(0, height, dtype=x.dtype, device=x.device),
                             torch.arange(0, width, dtype=x.dtype, device=x.device)])  # (H *W)

    # mh, mw = torch.meshgrid([torch.arange(0, height, dtype=x.dtype),
    #                              torch.arange(0, width, dtype=x.dtype)])  # (H *W)
    mh = mh.reshape(1, 1, height, width).repeat(bs, ndisp, 1, 1)
    mw = mw.reshape(1, 1, height, width).repeat(bs, ndisp, 1, 1)  # (B, D, H, W)

    cur_disp_coords_y = mh
    cur_disp_coords_x = mw - disp_range_samples
    # print('##############3333333', mw, cur_disp_coords_x)

    # print("cur_disp", cur_disp, cur_disp.shape if not isinstance(cur_disp, float) else 0)
    # print("cur_disp_coords_x", cur_disp_coords_x, cur_disp_coords_x.shape)

    coords_x = cur_disp_coords_x / ((width - 1.0) / 2.0) - 1.0  # trans to -1 - 1
    coords_y = cur_disp_coords_y / ((height - 1.0) / 2.0) - 1.0
    grid = torch.stack([coords_x, coords_y], dim=4)  # (B, D, H, W, 2)

    y_warped = F.grid_sample(y, grid.view(bs, ndisp * height, width, 2), mode='bilinear',
                             padding_mode='zeros', align_corners=True).view(bs, channels, ndisp, height,
                                                                            width)  # (B, C, D, H, W)

    # a littel difference, no zeros filling
    x_warped = x.unsqueeze(2).repeat(1, 1, ndisp, 1, 1)  # (B, C, D, H, W)
    # x_warped = x_warped.transpose(0, 1) #(C, B, D, H, W)
    #     #x1 = x2 + d >= d
    # x_warped[:, mw < disp_range_samples] = 0
    # x_warped = x_warped.transpose(0, 1) #(B, C, D, H, W)

    return y_warped, x_warped


def groupwise_correlation_4D(fea1, fea2, num_groups):
    B, C, D, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, D, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, D, H, W)
    return cost

class FocalModulation(nn.Module):
    def __init__(self, dim, focal_window=3, focal_level=2, focal_factor=2, bias=True, proj_drop=0.,
                 use_postln_in_modulation=False, normalize_modulator=False):
        super().__init__()

        self.dim = dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.use_postln_in_modulation = use_postln_in_modulation
        self.normalize_modulator = normalize_modulator

        self.f_linear = nn.Conv2d(dim, 2 * dim + (self.focal_level + 1), kernel_size=1, bias=bias)
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=bias)

        self.act = nn.GELU()
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)
        self.focal_layers = nn.ModuleList()

        self.kernel_sizes = []
        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1,
                              groups=dim, padding=kernel_size // 2, bias=False),
                    nn.GELU(),
                )
            )
            self.kernel_sizes.append(kernel_size)
        if self.use_postln_in_modulation:
            self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, H, W, C)
        """
        C = x.shape[1]

        # pre linear projection
        x = self.f_linear(x).contiguous()
        q, ctx, gates = torch.split(x, (C, C, self.focal_level + 1), 1)

        # context aggreation
        ctx_all = 0.0
        for l in range(self.focal_level):
            ctx = self.focal_layers[l](ctx)
            ctx_all = ctx_all + ctx * gates[:, l:l + 1]
        ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        ctx_all = ctx_all + ctx_global * gates[:, self.focal_level:]

        # normalize context
        if self.normalize_modulator:
            ctx_all = ctx_all / (self.focal_level + 1)

        # focal modulation
        x_out = q * self.h(ctx_all)
        x_out = x_out.contiguous()
        if self.use_postln_in_modulation:
            x_out = self.ln(x_out)

        # post linear porjection
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out
        
class Propagation(nn.Module):
    def __init__(self):
        super(Propagation, self).__init__()
        self.replicationpad = nn.ReplicationPad2d(1)

    def forward(self, disparity_samples):

        one_hot_filter = torch.zeros(5, 1, 3, 3, device=disparity_samples.device).float()
        one_hot_filter[0, 0, 0, 0] = 1.0
        one_hot_filter[1, 0, 1, 1] = 1.0
        one_hot_filter[2, 0, 2, 2] = 1.0
        one_hot_filter[3, 0, 2, 0] = 1.0
        one_hot_filter[4, 0, 0, 2] = 1.0
        disparity_samples = self.replicationpad(disparity_samples)
        aggregated_disparity_samples = F.conv2d(disparity_samples,
                                                    one_hot_filter,padding=0)
                                                    
        return aggregated_disparity_samples
        

class Propagation_prob(nn.Module):
    def __init__(self):
        super(Propagation_prob, self).__init__()
        self.replicationpad = nn.ReplicationPad3d((1, 1, 1, 1, 0, 0))

    def forward(self, prob_volume):
        one_hot_filter = torch.zeros(5, 1, 1, 3, 3, device=prob_volume.device).float()
        one_hot_filter[0, 0, 0, 0, 0] = 1.0
        one_hot_filter[1, 0, 0, 1, 1] = 1.0
        one_hot_filter[2, 0, 0, 2, 2] = 1.0
        one_hot_filter[3, 0, 0, 2, 0] = 1.0
        one_hot_filter[4, 0, 0, 0, 2] = 1.0

        prob_volume = self.replicationpad(prob_volume)
        prob_volume_propa = F.conv3d(prob_volume, one_hot_filter,padding=0)


        return prob_volume_propa