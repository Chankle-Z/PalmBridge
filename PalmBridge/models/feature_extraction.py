"""
@FileName   feature_extraction
@Author  24
@Date    2024/1/26 22:34
@Version 1.0.0
freedom is the oxygen of the soul.
"""
import math

import torch
from torch import nn

from models.component.gabor import GaborConv2d
from models.component.squeeze_and_excitation import SEModule
from models.vit import ViT


def get_sequence_feature(feature_tensor, vit_floor_num):
    # torch.Size([batch_size, 64, 30, 30])  torch.Size([batch_size, 64, 14, 14])
    feature_tensor_for_channel = torch.softmax(feature_tensor, dim=1)

    # torch.Size([batch_size, vit_floor_num, 30, 30])  torch.Size([batch_size, vit_floor_num, 14, 14])
    feature_tensor_for_channel_front = feature_tensor_for_channel[:, :vit_floor_num, :, :]
    feature_tensor_for_channel_back = feature_tensor_for_channel[:, (vit_floor_num * -1):, :, :]

    # torch.Size([batch_size, vit_floor_num * 2, 30, 30])  torch.Size([batch_size, vit_floor_num * 2, 14, 14])
    feature_tensor = torch.cat((feature_tensor_for_channel_front, feature_tensor_for_channel_back), dim=1)

    return feature_tensor


class FeatureExtraction(nn.Module):

    def __init__(self, channel_in, filter_num, kernel_size, stride, padding, init_ratio, label_num, vit_floor_num,
                 channel_out=36):
        super(FeatureExtraction, self).__init__()

        self.channel_in = channel_in  #
        self.filter_num = filter_num  #
        self.kernel_size = kernel_size  #
        self.stride = stride  #
        self.padding = padding  #
        self.init_ratio = init_ratio  #
        self.label_num = label_num  #
        self.vit_floor_num = vit_floor_num  #
        self.channel_out = channel_out  #

        #
        self.gabor_conv2d_1 = GaborConv2d(channel_in=self.channel_in, channel_out=self.filter_num,
                                          kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                                          init_ratio=self.init_ratio)
        self.gabor_conv2d_2 = GaborConv2d(channel_in=self.filter_num, channel_out=self.filter_num,
                                          kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                                          init_ratio=self.init_ratio)

        # SE（Squeeze-and-Excitation
        self.squeeze_and_excitation = SEModule(channel=self.filter_num)

        #
        #
        self.conv_0 = nn.Conv2d(in_channels=self.filter_num, out_channels=64, kernel_size=5, stride=1, padding=0)
        self.conv_1 = nn.Conv2d(in_channels=self.filter_num, out_channels=64, kernel_size=5, stride=1, padding=0)

        #
        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=0)
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=0)

        #
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, feature_tensor):  # feature_tensor的torch.Size([batch_size, channel_in, 128, 128])
        #
        #    torch.Size([batch_size, 16, 64, 64])
        first_order_feature_tensor = self.gabor_conv2d_1(feature_tensor)
        #      torch.Size([batch_size, 16, 32, 32])
        second_order_feature_tensor = self.gabor_conv2d_2(first_order_feature_tensor)

        #
        # torch.Size([batch_size, 64, 30, 30])
        first_order_feature_tensor = self.process_block(first_order_feature_tensor, conv=self.conv_0)
        # torch.Size([batch_size, 64, 14, 14])
        second_order_feature_tensor = self.process_block(second_order_feature_tensor, conv=self.conv_1)

        #
        # torch.Size([batch_size, 32, 14, 14])
        f_order_feature_tensor = self.conv_2(first_order_feature_tensor)
        # torch.Size([batch_size, 32, 6, 6])
        s_order_feature_tensor = self.conv_3(second_order_feature_tensor)

        # torch.Size([batch_size, 32 * 14 * 14 + 32 * 6 * 6 = 7424])
        feature_tensor = torch.cat((f_order_feature_tensor.view(f_order_feature_tensor.shape[0], -1),
                                    s_order_feature_tensor.view(s_order_feature_tensor.shape[0], -1)), dim=1)

        # torch.Size([batch_size, vit_floor_num * 2, 30, 30])
        first_order_feature_tensor = get_sequence_feature(first_order_feature_tensor, self.vit_floor_num)
        # torch.Size([batch_size, vit_floor_num * 2, 14, 14])
        second_order_feature_tensor = get_sequence_feature(second_order_feature_tensor, self.vit_floor_num)

        return feature_tensor, first_order_feature_tensor, second_order_feature_tensor

    #
    def process_block(self, feature_tensor, conv):

        #
        # torch.Size([batch_size, 64, 64, 64])   二阶torch.Size([batch_size, 64, 32, 32])
        feature_tensor = self.squeeze_and_excitation(feature_tensor)

        #
        # torch.Size([batch_size, 64, 60, 60])  二阶torch.Size([batch_size, 64, 28, 28])
        feature_tensor = conv(feature_tensor)

        #
        # torch.Size([batch_size, 64, 60, 60])  二阶torch.Size([batch_size, 64, 28, 28])
        feature_tensor = torch.relu(feature_tensor)

        #
        # torch.Size([batch_size, 64, 30, 30])  二阶torch.Size([batch_size, 64, 14, 14])
        feature_tensor = self.max_pool(feature_tensor)

        return feature_tensor


'''  
may the force be with you.
@FileName   feature_extraction
Created by 24 on 2024/1/26.
'''
