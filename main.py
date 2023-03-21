import copy
import functools
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from equi_pers.equi2pers_v3 import equi2pers
from equi_pers.pers2equi_v3 import pers2equi
from torch.nn.modules import padding
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d

from .ViT import miniViT, layers
from .ViT.layers import PixelWiseDotProduct
from .blocks import Transformer_Block


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBnReLU_v2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU_v2, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, (kernel_size, kernel_size, 1),
                              stride=(stride, stride, 1), padding=(pad, pad, 0), bias=False, padding_mode='zeros')
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


def convert_conv(layer):
    for name, module in layer.named_modules():
        if name:
            try:
                sub_layer = getattr(layer, name)
                if isinstance(sub_layer, nn.Conv2d):
                    m = copy.deepcopy(sub_layer)
                    new_layer = nn.Conv3d(m.in_channels, m.out_channels,
                                          kernel_size=(m.kernel_size[0], m.kernel_size[1], 1),
                                          stride=(m.stride[0], m.stride[1], 1), padding=(m.padding[0], m.padding[1], 0),
                                          padding_mode='zeros', bias=False)
                    new_layer.weight.data.copy_(m.weight.data.unsqueeze(-1))
                    if m.bias is not None:
                        new_layer.bias.data.copy_(m.bias.data)
                    layer._modules[name] = copy.deepcopy(new_layer)
            except AttributeError:
                name = name.split('.')[0]
                sub_layer = getattr(layer, name)
                sub_layer = convert_conv(sub_layer)
                layer.__setattr__(name=name, value=sub_layer)
    return layer


def convert_bn(layer):
    for name, module in layer.named_modules():
        if name:
            try:
                sub_layer = getattr(layer, name)
                if isinstance(sub_layer, nn.BatchNorm2d):
                    m = copy.deepcopy(sub_layer)
                    new_layer = nn.BatchNorm3d(m.num_features)
                    new_layer.weight.data.copy_(m.weight.data)
                    new_layer.bias.data.copy_(m.bias.data)
                    new_layer.running_mean.data.copy_(m.running_mean.data)
                    new_layer.running_var.data.copy_(m.running_var.data)
                    layer._modules[name] = copy.deepcopy(new_layer)
            except AttributeError:
                name = name.split('.')[0]
                sub_layer = getattr(layer, name)
                sub_layer = convert_bn(sub_layer)
                layer.__setattr__(name=name, value=sub_layer)
    return layer


class Transformer_cascade(nn.Module):
    def __init__(self, emb_dims, num_patch, depth, num_heads):
        super(Transformer_cascade, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(emb_dims, eps=1e-6)
        self.pos_emb = nn.Parameter(torch.zeros(1, num_patch, emb_dims))
        nn.init.trunc_normal_(self.pos_emb, std=.02)
        for _ in range(depth):
            layer = Transformer_Block(emb_dims, num_heads=num_heads)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x):
        hidden_states = x + self.pos_emb
        for i, layer_block in enumerate(self.layer):
            hidden_states = layer_block(hidden_states)

        encoded = self.encoder_norm(hidden_states)
        return encoded


class Holistic_network(nn.Module):
    def __init__(self):
        super(Holistic_network, self).__init__()
        pretrain_model = torchvision.models.resnet34(pretrained=True)

        encoder = pretrain_model

        self.conv1 = encoder.conv1
        self.bn1 = encoder.bn1
        self.relu = nn.ReLU(True)
        self.layer1 = encoder.layer1  # 64
        self.layer2 = encoder.layer2  # 128
        self.layer3 = encoder.layer3  # 256
        self.layer4 = encoder.layer4  # 512

        self.down = nn.Conv2d(512, 512 // 4, kernel_size=1, stride=1, padding=0)

        self.de_conv0_0 = ConvBnReLU(512 + 128, 256, kernel_size=3, stride=1)
        self.de_conv0_1 = ConvBnReLU(256 + 256, 128, kernel_size=3, stride=1)
        self.de_conv1_0 = ConvBnReLU(128, 128, kernel_size=3, stride=1)
        self.de_conv1_1 = ConvBnReLU(128 + 128, 64, kernel_size=3, stride=1)
        self.de_conv2_0 = ConvBnReLU(64, 64, kernel_size=3, stride=1)
        self.de_conv2_1 = ConvBnReLU(64 + 64, 64, kernel_size=3, stride=1)
        self.de_conv3_0 = ConvBnReLU(64, 64, kernel_size=3, stride=1)
        self.de_conv3_1 = ConvBnReLU(64 + 64, 32, kernel_size=3, stride=1)
        self.de_conv4_0 = ConvBnReLU(32, 32, kernel_size=3, stride=1)

        self.transformer = Transformer_cascade(128, 16 * 32, depth=6, num_heads=4)

    def forward(self, rgb):
        bs, c, erp_h, erp_w = rgb.shape
        conv1 = self.relu(self.bn1(self.conv1(rgb)))
        pool = F.max_pool2d(conv1, kernel_size=3, stride=2, padding=1)

        layer1 = self.layer1(pool)

        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        layer4_reshape = self.down(layer4)

        layer4_reshape = layer4_reshape.permute(0, 2, 3, 1).reshape(bs, 16 * 32, -1)
        layer4_reshape = self.transformer(layer4_reshape)

        layer4_reshape = layer4_reshape.permute(0, 2, 1).reshape(bs, -1, 16, 32)
        layer4 = torch.cat([layer4, layer4_reshape], 1)

        up = F.interpolate(layer4, size=(layer3.shape[-2], layer3.shape[-1]), mode='bilinear', align_corners=False)

        de_conv0_0 = self.de_conv0_0(up)
        concat = torch.cat([de_conv0_0, layer3], 1)
        de_conv0_1 = self.de_conv0_1(concat)

        up = F.interpolate(de_conv0_1, size=(layer2.shape[-2], layer2.shape[-1]), mode='bilinear', align_corners=False)
        de_conv1_0 = self.de_conv1_0(up)
        concat = torch.cat([de_conv1_0, layer2], 1)
        de_conv1_1 = self.de_conv1_1(concat)

        up = F.interpolate(de_conv1_1, size=(layer1.shape[-2], layer1.shape[-1]), mode='bilinear', align_corners=False)
        de_conv2_0 = self.de_conv2_0(up)
        concat = torch.cat([de_conv2_0, layer1], 1)
        de_conv2_1 = self.de_conv2_1(concat)

        up = F.interpolate(de_conv2_1, size=(conv1.shape[-2], conv1.shape[-1]), mode='bilinear', align_corners=False)
        de_conv3_0 = self.de_conv3_0(up)
        concat = torch.cat([de_conv3_0, conv1], 1)
        de_conv3_1 = self.de_conv3_1(concat)

        de_conv4_0 = self.de_conv4_0(de_conv3_1)

        return de_conv4_0


class Regional_network(nn.Module):
    def __init__(self):
        super(Regional_network, self).__init__()
        pretrain_model = torchvision.models.resnet34(pretrained=True)

        encoder = convert_conv(pretrain_model)
        encoder = convert_bn(encoder)

        self.conv1 = encoder.conv1
        self.bn1 = encoder.bn1

        self.relu = nn.ReLU(True)
        self.layer1 = encoder.layer1  # 64
        self.layer2 = encoder.layer2  # 128
        self.layer3 = encoder.layer3  # 256
        self.layer4 = encoder.layer4  # 512

        self.de_conv0 = ConvBnReLU_v2(128, 64, kernel_size=3, stride=1)
        self.de_conv1 = ConvBnReLU_v2(64, 64, kernel_size=3, stride=1)
        self.de_conv2 = ConvBnReLU_v2(64, 64, kernel_size=3, stride=1)
        self.de_conv3 = ConvBnReLU_v2(64, 128, kernel_size=3, stride=1)

        self.down = nn.Conv3d(512, 512 // 4, kernel_size=(4, 4, 1), stride=(4, 4, 1), padding=0)

        self.tangent_layer = miniViT.tangent_ViT(128, n_query_channels=128, patch_size=4,
                                                 dim_out=200,
                                                 embedding_dim=128, norm='linear')

        self.mlp_points = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

    def forward(self, tangent, center_point, uv):
        bs, c, patch_h, patch_w, n_patch = tangent.shape

        device = tangent.device
        rho = torch.ones((uv.shape[0], 1, patch_h // 4, patch_w // 4), dtype=torch.float32, device=device)

        center_points = center_point.to(device)
        center_points = center_points.reshape(-1, 2, 1, 1).repeat(1, 1, patch_h // 4, patch_w // 4)

        new_xyz = torch.cat([center_points, rho, center_points], 1)
        point_feat = self.mlp_points(new_xyz.contiguous())
        point_feat = point_feat.permute(1, 2, 3, 0).unsqueeze(0)

        conv1 = self.relu(self.bn1(self.conv1(tangent)))
        pool = F.max_pool3d(conv1, kernel_size=(3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0))

        layer1 = self.layer1(pool)

        layer1 = layer1 + point_feat
        layer2 = self.layer2(layer1)

        layer3 = self.layer3(layer2)

        layer4 = self.layer4(layer3)

        layer4_reshape = self.down(layer4)

        tangent_feature_set = layer4_reshape

        layer4_reshape = layer4_reshape.permute(0, 2, 3, 4, 1).flatten(-2).permute(0, 3, 1, 2)
        up = F.interpolate(layer4_reshape, size=(layer4.shape[-3], layer4.shape[-2]), mode='bilinear',
                           align_corners=False)
        up = up.permute(0, 2, 3, 1).reshape(bs, layer4.shape[-3], layer4.shape[-2], -1, n_patch).permute(0, 3, 1, 2, 4)

        deconv_layer4 = self.de_conv0(up)
        deconv_layer4 = deconv_layer4.permute(0, 2, 3, 4, 1).flatten(-2).permute(0, 3, 1, 2)
        up = F.interpolate(deconv_layer4, size=(layer3.shape[-3], layer3.shape[-2]), mode='bilinear',
                           align_corners=False)
        up = up.permute(0, 2, 3, 1).reshape(bs, layer3.shape[-3], layer3.shape[-2], -1, n_patch).permute(0, 3, 1, 2, 4)

        deconv_layer3 = self.de_conv1(up)
        deconv_layer3 = deconv_layer3.permute(0, 2, 3, 4, 1).flatten(-2).permute(0, 3, 1, 2)
        up = F.interpolate(deconv_layer3, size=(layer2.shape[-3], layer2.shape[-2]), mode='bilinear',
                           align_corners=False)
        up = up.permute(0, 2, 3, 1).reshape(bs, layer2.shape[-3], layer2.shape[-2], -1, n_patch).permute(0, 3, 1, 2, 4)

        deconv_layer2 = self.de_conv2(up)
        deconv_layer2 = deconv_layer2.permute(0, 2, 3, 4, 1).flatten(-2).permute(0, 3, 1, 2)
        up = F.interpolate(deconv_layer2, size=(layer1.shape[-3], layer1.shape[-2]), mode='bilinear',
                           align_corners=False)
        up = up.permute(0, 2, 3, 1).reshape(bs, layer1.shape[-3], layer1.shape[-2], -1, n_patch).permute(0, 3, 1, 2, 4)
        deconv_layer1 = self.de_conv3(up)

        bs, c, h, w, n_patch = deconv_layer1.shape

        tangent_embedding_set = []
        tangent_bin_set = []

        for i in range(n_patch):
            tangent_feature = deconv_layer1[..., i]
            tangent_bin, tangent_embedding = self.tangent_layer(tangent_feature)
            tangent_embedding_set.append(tangent_embedding)
            tangent_bin_set.append(tangent_bin)
        tangent_embedding_set = torch.stack(tangent_embedding_set, dim=-1)
        tangent_bin_set = torch.stack(tangent_bin_set, dim=-1)
        return tangent_bin_set, tangent_embedding_set, tangent_feature_set


class HRDFuse(nn.Module):
    def __init__(self, nrows=4, npatches=18, patch_size=(128, 128), fov=(80, 80), nbins=100, min_val=0.1, max_val=10, ):
        self.num_classes = nbins
        self.min_val = min_val
        self.max_val = max_val
        self.nrows = nrows
        self.npatches = npatches
        self.patch_size = patch_size
        self.fov = fov
        super(HRDFuse, self).__init__()

        self.holistic_network = Holistic_network()
        self.regional_network = Regional_network()

        self.pred = nn.Conv3d(32, 1, (3, 3, 1), 1, padding=(1, 1, 0), padding_mode='zeros')
        self.weight_pred = nn.Conv3d(32, 1, (3, 3, 1), 1, padding=(1, 1, 0), padding_mode='zeros')
        self.min_depth = 0.1
        self.max_depth = 10.0

        self.conv_out_holistic = nn.Sequential(nn.Conv2d(128, self.num_classes, kernel_size=1, stride=1, padding=0),
                                               nn.Softmax(dim=1))

        self.conv_out_regional = nn.Sequential(nn.Conv2d(128, self.num_classes, kernel_size=1, stride=1, padding=0),
                                               nn.Softmax(dim=1))

        self.adaptive_bins_layer = miniViT.mViT(32, n_query_channels=128, patch_size=8,
                                                dim_out=200,
                                                embedding_dim=128, norm='linear')

        self.wise_dot_layer = PixelWiseDotProduct()

        self.w = nn.Parameter(torch.ones(2))

    def calculate_index_map(self, regional_feature_set, feature_holistic):
        bs, c, _, _, N = regional_feature_set.shape
        regional_feature = regional_feature_set.reshape(bs, c, N).permute(0, 2, 1)

        regional_feature_normed = torch.norm(regional_feature, p=2, dim=-1).unsqueeze(-1).unsqueeze(-1)

        holistic_feature_normed = torch.norm(feature_holistic, p=2, dim=1).unsqueeze(1)

        similarity_map = torch.einsum('bne, behw -> bnhw', regional_feature, feature_holistic)

        similarity_map = similarity_map / regional_feature_normed
        similarity_map = similarity_map / holistic_feature_normed

        similarity_max_map, similarity_index_map = torch.max(similarity_map, dim=1, keepdim=True)

        one_hot = torch.FloatTensor(similarity_index_map.shape[0], regional_feature.shape[1],
                                    similarity_index_map.shape[2], similarity_index_map.shape[3]).zero_().to(
            regional_feature.device)
        similarity_index_map = one_hot.scatter_(1, similarity_index_map, 1)

        return similarity_index_map

    def forward(self, rgb, confidence=True):
        bs, _, erp_h, erp_w = rgb.shape
        device = rgb.device
        patch_h, patch_w = pair(self.patch_size)

        high_res_patch, _, _, _ = equi2pers(rgb, self.fov, self.nrows, patch_size=self.patch_size)
        _, xyz, uv, center_points = equi2pers(rgb, self.fov, self.nrows, patch_size=(patch_h, patch_w))

        center_points = center_points.to(device)

        holistic_feature = self.holistic_network(rgb)

        regional_bin_set, regional_embedding_set, regional_feature_set = self.regional_network(high_res_patch,
                                                                                               center_points, uv)
        regional_embedding_set = torch.mean(regional_embedding_set, dim=1, keepdim=False)

        _, holistic_bin_widths, holistic_range_attention_map, holistic_queries, holistic_feature_map = \
            self.adaptive_bins_layer(holistic_feature)

        holistic_range_attention_map = F.interpolate(holistic_range_attention_map, (erp_h, erp_w), mode='bilinear')

        similarity_index_map = self.calculate_index_map(regional_feature_set, holistic_feature_map)

        regional_bin_map = torch.einsum('ben, bnhw -> behw', regional_bin_set, similarity_index_map)
        regional_bin_map = F.interpolate(regional_bin_map, (erp_h, erp_w), mode='bilinear')

        regional_query_map = torch.einsum('ben, bnhw -> behw', regional_embedding_set, similarity_index_map)
        regional_range_attention_map = torch.einsum('bse, behw -> bshw', holistic_queries, regional_query_map)
        regional_range_attention_map = F.interpolate(regional_range_attention_map, (erp_h, erp_w), mode='bilinear')

        holistic_prob_map = self.conv_out_holistic(holistic_range_attention_map)

        regional_prob_map = self.conv_out_regional(regional_range_attention_map)

        holistic_bin_widths = nn.functional.pad((self.max_val - self.min_val) * holistic_bin_widths, (1, 0),
                                                mode='constant', value=self.min_val)

        regional_bin_map = nn.functional.pad((self.max_val - self.min_val) * regional_bin_map, (0, 0, 0, 0, 1, 0),
                                             mode='constant',
                                             value=self.min_val)

        holistic_bin_edges = torch.cumsum(holistic_bin_widths, dim=1)

        regional_bin_edges = torch.cumsum(regional_bin_map, dim=1)

        holistic_bin_centers = 0.5 * (holistic_bin_edges[:, :-1] + holistic_bin_edges[:, 1:])
        n_erp, dout_erp = holistic_bin_centers.size()
        holistic_bin_centers = holistic_bin_centers.view(n_erp, dout_erp, 1, 1)

        regional_bin_centers = 0.5 * (regional_bin_edges[:, :-1] + regional_bin_edges[:, 1:])

        regional_prediction = torch.sum(regional_prob_map * regional_bin_centers, dim=1, keepdim=True)
        holistic_prediction = torch.sum(holistic_prob_map * holistic_bin_centers, dim=1, keepdim=True)

        w0 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w1 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        pred = w0 * holistic_prediction + w1 * regional_prediction

        return pred, holistic_bin_edges, regional_prediction, holistic_prediction, regional_query_map, holistic_queries


if __name__ == "__main__":
    net = HRDFuse(4, 18)
    input = torch.randn((3, 3, 512, 1024), dtype=torch.float32)
    output, bin_edges, _, _, _, _ = net(input)
    print(output.shape)
