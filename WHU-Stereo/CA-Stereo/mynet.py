import torch
import torch.nn as nn
import torch.nn.functional as F
from computation import Estimation

from refinment import *
from feature import *
from aggregation import *
from submodule import *
import time


class encoder_disp(nn.Module):
    def __init__(self,inchannel):
        super(encoder_disp, self).__init__()
        self.stem_2 = nn.Sequential(
            BasicConv_IN(inchannel, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU()
        )
        self.stem_4 = nn.Sequential(
            BasicConv_IN(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48), nn.ReLU()
        )
        self.stem_8 = nn.Sequential(
            BasicConv_IN(48, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48), nn.ReLU()
        )
        self.stem_16 = nn.Sequential(
            BasicConv_IN(48, 64, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(64), nn.ReLU()
        )        
    def forward(self, disp, gx, gy):
        disparity = disp.unsqueeze(1)
        concat = torch.cat([disparity, gx, gy], dim=1)
        stem_2x = self.stem_2(concat)
        stem_4x = self.stem_4(stem_2x)
        stem_8x = self.stem_8(stem_4x)
        stem_16x = self.stem_16(stem_8x)
        return [stem_2x, stem_4x, stem_8x, stem_16x]      
    
    
class Refine(nn.Module):
    def __init__(self):
        super(Refine, self).__init__()
        self.stem16_8 = nn.Sequential(
            BasicConv_IN(64, 48, deconv=True,kernel_size=4, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48), nn.ReLU()
        )
        self.stem8_4 = nn.Sequential(
            BasicConv_IN(48, 48, deconv=True,kernel_size=4, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48), nn.ReLU()
        )
        self.stem4_2 = nn.Sequential(
            BasicConv_IN(48, 32,deconv=True, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU()
        )
        self.stem2_f = nn.Sequential(
            BasicConv_IN(32, 32,deconv=True, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU()
        )  
        self.conv = nn.Conv2d(in_channels=32,  out_channels=1, kernel_size=3, stride=1, padding=1)
    def forward( self, disp,d2,d4,d8,d16, l2, l4,l8,l16):
        disp = disp.unsqueeze(1)
        # 残差学习
        f16_8 = self.stem16_8(d16+l16)
        f8_4 = self.stem8_4(f16_8 +d8 +l8 )
        f4_2 = self.stem4_2(f8_4 +d4 +l4 )
        f2_f = self.stem2_f(f4_2 +d2 +l2 )
        f2_f =self.conv( f2_f)

        # 生成最终的视差图
        disp_final =f2_f + disp
        return disp_final
                
              
        
class CAStereo(nn.Module):
    def __init__(self, mindisp, maxdisp):
        super().__init__()
        self.max_disp = maxdisp
        self.min_disp = mindisp
        self.feature = Feature()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(2 * torch.ones(1))
        self.propagation1 = Propagation()
        self.propagation_prob1 = Propagation_prob()
        self.propagation2 = Propagation()
        self.propagation_prob2 = Propagation_prob()
        
        self.encoder = encoder_disp(3)
        self.drefine = Refine()
        
        
        
        self.stem_2 = nn.Sequential(
            BasicConv_IN(1, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU()
        )
        self.stem_4 = nn.Sequential(
            BasicConv_IN(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48), nn.ReLU()
        )
        self.stem_8 = nn.Sequential(
            BasicConv_IN(48, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48), nn.ReLU()
        )
        self.stem_16 = nn.Sequential(
            BasicConv_IN(48, 64, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(64), nn.ReLU()
        )        

        self.corr_stem0 = BasicConv(16, 16, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.corr_stem1 = BasicConv(16, 16, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.corr_stem2 = BasicConv(16, 16, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.corr_feature_att4 = FeatureAtt(16, 96)
        self.corr_feature_att4_8 = FeatureAtt(16, 112)
        self.corr_feature_att16 = FeatureAtt(16, 256)
        self.cost_agg0 = hourglass(16)
        self.cost_agg1 = hourglass(16)
        self.cost_agg2 = hourglass(16)
        self.cost_agg3 = hourglass(16)
        self.cost_agg4 = hourglass(16)
        
        self.cost_agg0_1 = hourglass(16)
        self.cost_agg1_1 = hourglass(16)
        self.cost_agg2_1 = hourglass(16)
        
        self.estimator0 = Estimation(min_disp=self.min_disp // 4, max_disp=self.max_disp // 4)
        self.estimator1 = Estimation(min_disp=self.min_disp // 8, max_disp=self.max_disp // 8)
        self.estimator2 = Estimation(min_disp=self.min_disp // 16, max_disp=self.max_disp // 16)
        self.conv1 = nn.Conv3d(
            in_channels=16,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,  # 'same' padding
            bias=True
        )
        self.conv2 = nn.Conv3d(
            in_channels=16,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,  # 'same' padding
            bias=True
        )
        self.conv3 = nn.Conv3d(
            in_channels=16,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,  # 'same' padding
            bias=True
        )
        self.spx = nn.Sequential(nn.ConvTranspose2d(2 * 32, 9, kernel_size=4, stride=2, padding=1), )
        self.spx_2 = Conv2x(24, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv(96, 24, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 24, 3, 1, 1, bias=False),
            nn.BatchNorm2d(24), nn.ReLU())

        self.fusion1 = FeatureFusion(16)
        self.fusion2 = FeatureFusion(16)
        self.fusion1_1 = FeatureFusion(16)
        self.fusion2_1 = FeatureFusion(16)
            

    def forward(self, image1, image2, dx, dy):
        features_left = self.feature(image1)
        features_right = self.feature(image2)
        stem_2x = self.stem_2(image1)
        stem_4x = self.stem_4(stem_2x)
        stem_8x = self.stem_8(stem_4x)
        stem_16x = self.stem_16(stem_8x)
        
        stem_2y = self.stem_2(image2)
        stem_4y = self.stem_4(stem_2y)
        stem_8y = self.stem_8(stem_4y)
        stem_16y = self.stem_16(stem_8y)

        features_left[0] = torch.cat((features_left[0], stem_4x), 1)
        features_right[0] = torch.cat((features_right[0], stem_4y), 1)

        features_left[1] = torch.cat((features_left[1], stem_8x), 1)
        features_right[1] = torch.cat((features_right[1], stem_8y), 1)
        
        features_left[2] = torch.cat((features_left[2], stem_16x), 1)
        features_right[2] = torch.cat((features_right[2], stem_16y), 1)
        
        gwc_volume0 = build_gwc_volume(features_left[0], features_right[0], self.min_disp // 4, self.max_disp // 4,
                                       16)
        gwc_volume1 = build_gwc_volume(features_left[1], features_right[1], self.min_disp // 8, self.max_disp // 8,
                                       16)
        gwc_volume2 = build_gwc_volume(features_left[2], features_right[2], self.min_disp // 16, self.max_disp // 16,
                                       16)

        gwc_volume2 = self.corr_stem2(gwc_volume2)
        gwc_volume2 = self.corr_feature_att16(gwc_volume2, features_left[2])
        context_encoding_volume2 = self.cost_agg0(gwc_volume2)
        context_encoding_volume2_1 = self.cost_agg2_1(context_encoding_volume2)
        att1 = self.conv3(context_encoding_volume2)
        a1, disparity2 = self.estimator2(context_encoding_volume2_1)
        att1 = F.interpolate(att1, [self.max_disp//8-self.min_disp // 8, image1.size()[2]//8, image1.size()[3]//8], mode='trilinear')
        
        pred_att = torch.squeeze(att1, 1)
        pred_att_prob = F.softmax(pred_att, dim=1)
        pred_att = disparity_regression(pred_att_prob, self.min_disp // 8, self.max_disp // 8)
        pred_variance = disparity_variance(pred_att_prob,  self.min_disp // 8, self.max_disp // 8, pred_att.unsqueeze(1))
        pred_variance = self.beta + self.gamma * pred_variance
        pred_variance = torch.sigmoid(pred_variance)
        pred_variance_samples = self.propagation1(pred_variance)
        disparity_samples = self.propagation1(pred_att.unsqueeze(1))
        right_feature_x8, left_feature_x8 = SpatialTransformer_grid(stem_8x, stem_8y, disparity_samples)
        disparity_sample_strength = (left_feature_x8 * right_feature_x8).mean(dim=1)
        disparity_sample_strength = torch.softmax(disparity_sample_strength * pred_variance_samples, dim=1)
        att_weights = self.propagation_prob1(att1)
        att_weights = att_weights * disparity_sample_strength.unsqueeze(2)
        att_weights = torch.sum(att_weights, dim=1, keepdim=True)
        att_weights_prob = F.softmax(att_weights, dim=2)
        _, ind = att_weights_prob.sort(2, True)
        k = 24
        ind_k = ind[:, :, :k]
        ind_k = ind_k.sort(2, False)[0]
        att_topk = torch.gather(att_weights_prob, 2, ind_k)
        disparity_sample_topk = ind_k.squeeze(1).float()
        
        gwc_volume1 = att_topk * gwc_volume1
        gwc_volume1 = self.corr_stem1(gwc_volume1)
        gwc_volume1 = self.corr_feature_att4_8(gwc_volume1, features_left[1])
        
        context_encoding_volume1 = self.cost_agg1(gwc_volume1)
        agg_cost1 = self.cost_agg2(context_encoding_volume1)
        fusion_cost1 = self.fusion1(context_encoding_volume2_1, agg_cost1)
        agg_fusion_cost1 = self.cost_agg1_1(fusion_cost1)
        att2 = self.conv3(context_encoding_volume1)
        a2,disparity1 = self.estimator1(agg_fusion_cost1)
        
        att2 = F.interpolate(att2, [self.max_disp//4-self.min_disp // 4, image1.size()[2]//4, image1.size()[3]//4], mode='trilinear')
        pred_att1 = torch.squeeze(att2, 1)
        pred_att_prob1 = F.softmax(pred_att1, dim=1)
        pred_att1 = disparity_regression(pred_att_prob1, self.min_disp // 4, self.max_disp// 4)
        pred_variance1 = disparity_variance(pred_att_prob1, self.min_disp // 4, self.max_disp // 4, pred_att1.unsqueeze(1))
        pred_variance1 = self.beta + self.gamma * pred_variance1
        pred_variance1 = torch.sigmoid(pred_variance1)
        pred_variance_samples1 = self.propagation2(pred_variance1)
        disparity_samples1 = self.propagation2(pred_att1.unsqueeze(1))
        right_feature_x4, left_feature_x4 = SpatialTransformer_grid(stem_4x, stem_4y, disparity_samples1)
        disparity_sample_strength1 = (left_feature_x4 * right_feature_x4).mean(dim=1)
        disparity_sample_strength1 = torch.softmax(disparity_sample_strength1 * pred_variance_samples1, dim=1)
        att_weights1 = self.propagation_prob2(att2)
        att_weights1 = att_weights1 * disparity_sample_strength1.unsqueeze(2)
        att_weights1 = torch.sum(att_weights1, dim=1, keepdim=True)
        att_weights_prob1 = F.softmax(att_weights1, dim=2)
        _, ind1 = att_weights_prob1.sort(2, True)
        k1 = 48
        ind_k1 = ind1[:, :, :k1]
        ind_k1 = ind_k1.sort(2, False)[0]
        att_topk1 = torch.gather(att_weights_prob1, 2, ind_k1)

        disparity_sample_topk1 = ind_k1.squeeze(1).float()
        
               
        gwc_volume0 =  att_topk1 * gwc_volume0
        gwc_volume0 = self.corr_stem0(gwc_volume0)
        gwc_volume0 = self.corr_feature_att4(gwc_volume0, features_left[0])
        context_encoding_volume0 = self.cost_agg3(gwc_volume0)
        agg_cost2 = self.cost_agg4(context_encoding_volume0)
        fusion_cost2 = self.fusion2(agg_fusion_cost1, agg_cost2)
        agg_fusion_cost2 = self.cost_agg0_1(fusion_cost2)
        _, disparity0 = self.estimator0(agg_fusion_cost2)


        xspx = self.spx_4(features_left[0])
        xspx = self.spx_2(xspx, stem_2x)
        spx_pred = self.spx(xspx)
        spx_pred = F.softmax(spx_pred, 1)
        scale_factor = image1.size(2) / disparity0.size(2)
        updisparity0 = context_upsample(disparity0, spx_pred)
        updisparity0 = updisparity0 * scale_factor
        
        
        disp_encoder = self.encoder(updisparity0,dx,dy)
        
        disp_final = self.drefine(updisparity0,disp_encoder[0],disp_encoder[1],disp_encoder[2],disp_encoder[3],stem_2x,stem_4x,stem_8x,stem_16x)
        
        return [disparity2, disparity1, disparity0, disp_final]