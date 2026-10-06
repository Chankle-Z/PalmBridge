"""
@FileName   net
@Author  24
@Date    2024/1/22 23:02
@Version 1.0.0
freedom is the oxygen of the soul.
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.vit import ViT
from models.component.arcface import ArcMarginProduct
from models.feature_extraction import FeatureExtraction


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # codebook: (num_embeddings, embedding_dim)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def forward(self, z):
        B, D = z.shape
        assert D == self.embedding_dim, f"输入维度 {D} must equal embedding_dim={self.embedding_dim}"

        # (B, D) -> (B, 1)
        distances = (
            torch.sum(z ** 2, dim=1, keepdim=True)
            - 2 * torch.matmul(z, self.embedding.weight.t())
            + torch.sum(self.embedding.weight ** 2, dim=1)
        )  # (B, num_embeddings)

        encoding_indices = torch.argmin(distances, dim=1)  # (B,)
        quantized = self.embedding(encoding_indices)  # (B, D)

        # 损失
        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # straight-through
        quantized = z + (quantized - z).detach()

        return quantized, loss

class VectorQuantizerLinInde(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, ortho_weight=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.ortho_weight = ortho_weight

        # codebook: (num_embeddings, embedding_dim)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def forward(self, z):
        B, D = z.shape
        assert D == self.embedding_dim, f"输入维度 {D} must equal embedding_dim={self.embedding_dim}"

        # (B, D) -> (B, num_embeddings)
        distances = (
            torch.sum(z ** 2, dim=1, keepdim=True)
            - 2 * torch.matmul(z, self.embedding.weight.t())
            + torch.sum(self.embedding.weight ** 2, dim=1)
        )

        encoding_indices = torch.argmin(distances, dim=1)  # (B,)
        quantized = self.embedding(encoding_indices)  # (B, D)

        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss

        W = F.normalize(self.embedding.weight, p=2, dim=1)   # (N, D)
        sim = torch.matmul(W, W.t())                         # (N, N)
        ortho_loss = ((sim - torch.eye(self.num_embeddings, device=z.device))**2).mean()

        loss = vq_loss + self.ortho_weight * ortho_loss

        # straight-through
        quantized = z + (quantized - z).detach()

        return quantized, loss


class SFNetVq(torch.nn.Module):

    def __init__(self, num_classes, vit_floor_num, weight=0.8):
        super(SFNetVq, self).__init__()

        self.label_num = num_classes
        self.vit_floor_num = vit_floor_num

        self.competitive_feature_extraction = FeatureExtraction(channel_in=1, filter_num=36,
                                                                kernel_size=17, stride=2, padding=17 // 2,
                                                                init_ratio=0.5, label_num=self.label_num,
                                                                vit_floor_num=self.vit_floor_num)

        # ViT
        self.vit_0 = ViT(image_size=30, patch_size=5, channels=self.vit_floor_num * 2, num_classes=self.label_num,
                       depth=2, heads=16, dim=128, dim_for_head=64, dim_for_mlp=256, dropout=0.1, emb_dropout=0.1)
        self.vit_1 = ViT(image_size=14, patch_size=2, channels=self.vit_floor_num * 2, num_classes=self.label_num,
                         depth=2, heads=16, dim=128, dim_for_head=64, dim_for_mlp=256, dropout=0.1, emb_dropout=0.1)

        self.quantizer_tensor = VectorQuantizerLinInde(512,7424)
        self.quantizer_vit = VectorQuantizerLinInde(512,11136)

        self.fully_connection_1 = torch.nn.Linear(7424, 2048)
        self.fully_connection_2 = torch.nn.Linear(2048, 1024)
        self.fully_connection_for_vit_1 = torch.nn.Linear(11136, 4096)
        self.fully_connection_for_vit_2 = torch.nn.Linear(4096, 1024)

        self.weight = weight

        self.dropout = torch.nn.Dropout(p=0.5)

        # ArcFace（Angular Margin Loss）
        self.arcface = ArcMarginProduct(in_features=1024, out_features=self.label_num)

    def forward(self, feature_tensor, target=None):  # feature_tensor的torch.Size([batch_size, 1, 128, 128])
        # torch.Size([batch_size, 1024])
        feature_tensor, vq_loss = self.processing(feature_tensor)

        feature_tensor = self.dropout(feature_tensor)

        # ArcFace（Angular Margin Loss）  torch.Size([batch_size, label_num])
        feature_tensor = self.arcface(feature_tensor, target)

        return feature_tensor, F.normalize(feature_tensor, dim=-1), vq_loss

    def getFeatureCode(self, feature_tensor):
        # torch.Size([batch_size, 1024])
        feature_tensor = self.processing_get(feature_tensor)

        # 返回对输入的特征张量进行L2范数归一化的结果
        return feature_tensor / torch.norm(feature_tensor, p=2, dim=1, keepdim=True)

    def processing(self, feature_tensor):
        # torch.Size([batch_size, 7424])
        # torch.Size([batch_size, vit_floor_num*2, 30, 30])
        # torch.Size([batch_size, vit_floor_num*2, 14, 14])
        feature_tensor, first_order_feature_tensor, second_order_feature_tensor \
            = self.competitive_feature_extraction(feature_tensor)

        # ViT
        # torch.Size([batch_size, num_patches + 1, dim])
        first_order_feature_tensor = self.vit_0(first_order_feature_tensor)
        # torch.Size([batch_size, num_patches + 1, dim])
        second_order_feature_tensor = self.vit_1(second_order_feature_tensor)

        # torch.Size([batch_size, 2 * (num_patches + 1), dim])
        feature_tensor_for_vit = torch.cat((first_order_feature_tensor, second_order_feature_tensor), dim=1)
        # torch.Size([batch_size, 11136])
        feature_tensor_for_vit = feature_tensor_for_vit.view(feature_tensor_for_vit.shape[0], -1)

        _, vq_loss_tensor = self.quantizer_tensor(feature_tensor)
        _, vq_loss_vit = self.quantizer_vit(feature_tensor_for_vit)

        feature_tensor = self.fully_connection_1(feature_tensor)  # torch.Size([batch_size, 2048])
        feature_tensor = self.fully_connection_2(feature_tensor)  # torch.Size([batch_size, 1024])
        # torch.Size([batch_size, 2048])
        feature_tensor_for_vit = self.fully_connection_for_vit_1(feature_tensor_for_vit)
        # torch.Size([batch_size, 1024])
        feature_tensor_for_vit = self.fully_connection_for_vit_2(feature_tensor_for_vit)

        # torch.Size([batch_size, 1024])
        feature_tensor = feature_tensor * self.weight + feature_tensor_for_vit * (1 - self.weight)
        vq_loss = vq_loss_tensor * self.weight + vq_loss_vit * (1 - self.weight)

        return feature_tensor, vq_loss

    def processing_get(self, feature_tensor):
        # torch.Size([batch_size, 7424])
        # torch.Size([batch_size, vit_floor_num*2, 30, 30])
        # torch.Size([batch_size, vit_floor_num*2, 14, 14])
        feature_tensor, first_order_feature_tensor, second_order_feature_tensor \
            = self.competitive_feature_extraction(feature_tensor)

        # ViT
        # torch.Size([batch_size, num_patches + 1, dim])
        first_order_feature_tensor = self.vit_0(first_order_feature_tensor)
        # torch.Size([batch_size, num_patches + 1, dim])
        second_order_feature_tensor = self.vit_1(second_order_feature_tensor)

        # torch.Size([batch_size, 2 * (num_patches + 1), dim])
        feature_tensor_for_vit = torch.cat((first_order_feature_tensor, second_order_feature_tensor), dim=1)
        # torch.Size([batch_size, 11136])
        feature_tensor_for_vit = feature_tensor_for_vit.view(feature_tensor_for_vit.shape[0], -1)

        # start_time = time.time()

        quantizer_tensor, _ = self.quantizer_tensor(feature_tensor)
        quantizer_vit_tensor, _ = self.quantizer_vit(feature_tensor_for_vit)

        feature_tensor = 0.7 * feature_tensor + 0.3 * quantizer_tensor
        feature_tensor_for_vit = 0.7 * feature_tensor_for_vit + 0.3 * quantizer_vit_tensor
        #
        # end_time = time.time()

        feature_tensor = self.fully_connection_1(feature_tensor)  # torch.Size([batch_size, 2048])
        feature_tensor = self.fully_connection_2(feature_tensor)  # torch.Size([batch_size, 1024])
        # torch.Size([batch_size, 2048])
        feature_tensor_for_vit = self.fully_connection_for_vit_1(feature_tensor_for_vit)
        # torch.Size([batch_size, 1024])
        feature_tensor_for_vit = self.fully_connection_for_vit_2(feature_tensor_for_vit)

        # torch.Size([batch_size, 1024])
        feature_tensor = feature_tensor * self.weight + feature_tensor_for_vit * (1 - self.weight)

        return feature_tensor


'''  
may the force be with you.
@FileName   net
Created by 24 on 2024/1/22.
'''
