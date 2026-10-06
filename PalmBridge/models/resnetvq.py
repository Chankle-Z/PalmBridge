import math
import time

import torch
import torch.nn as nn
from typing import Type, List, Optional, Callable, Union
import torch.nn.functional as F

def conv3x3(in_planes: int, out_planes: int, stride: int = 1,
            groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, 3, stride,
                     padding=dilation, groups=groups,
                     bias=False, dilation=dilation)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, 1, stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, **kwargs):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class Bottleneck(nn.Module):
    expansion: int = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, **kwargs):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class VectorQuantizerChannelWise(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # codebook: (num_embeddings, embedding_dim)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def forward(self, z):
        """
        z: (B, C, H, W)
        """
        B, C, H, W = z.shape
        D = H * W
        assert D == self.embedding_dim, f"H*W={D} must equal embedding_dim={self.embedding_dim}"

        z_flattened = z.view(B * C, -1)

        distances = (
            torch.sum(z_flattened ** 2, dim=1, keepdim=True)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
            + torch.sum(self.embedding.weight ** 2, dim=1)
        )

        encoding_indices = torch.argmin(distances, dim=1)  # (B*C,)

        quantized = self.embedding(encoding_indices)  # (B*C, D)

        quantized = quantized.view(B, C, H, W)

        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # straight-through estimator
        quantized = z + (quantized - z).detach()

        return quantized, loss

class VectorQuantizerChannelWiseLinIndep(nn.Module):
    def __init__(self, num_embeddings, embedding_dim,
                 commitment_cost=0.25, ortho_weight=0.1):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.ortho_weight = ortho_weight

        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def _make_linear_independent(self, W):
        K, D = W.shape
        if K <= D:
            return W
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        S = torch.clamp(S, min=1e-5)
        return U @ torch.diag_embed(S) @ Vh

    def forward(self, z):
        B, C, H, W_shape = z.shape
        D = H * W_shape
        assert D == self.embedding_dim
        z_flat = z.view(B * C, D)

        W = self._make_linear_independent(self.weight)

        distances = (
            (z_flat ** 2).sum(1, keepdim=True)
            - 2 * z_flat @ W.t()
            + (W ** 2).sum(1, keepdim=True).t()
        )

        indices = torch.argmin(distances, dim=1)
        quantized = F.embedding(indices, W).view(B, C, H, W_shape)

        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss

        W_norm = F.normalize(W, p=2, dim=1)
        sim = W_norm @ W_norm.t()
        eye = torch.eye(self.num_embeddings, device=W.device, dtype=W.dtype)
        ortho_loss = ((sim - eye) ** 2).mean()

        loss = vq_loss + self.ortho_weight * ortho_loss

        quantized = z + (quantized - z).detach()
        return quantized, loss

class ResNetVq(nn.Module):
    def __init__(self,
                 block: Type[Union[BasicBlock, Bottleneck]],
                 layers: List[int],
                 num_classes,
                 gray: bool = True):
        super().__init__()

        self.num_classes = num_classes
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1 if gray else 3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.quantizer_cb = VectorQuantizerChannelWiseLinIndep(256, 256)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))   # 128->64
        x = self.layer1(x)                       # 64
        x = self.layer2(x)                       # 32
        x = self.layer3(x)                       # 16
        _, vq_loss = self.quantizer_cb(x)
        x = self.layer4(x)                       # 8
        x = self.avgpool(x)                      # 1×1
        x = torch.flatten(x, 1)
        x = self.fc(x)
        fe = torch.cat((x,x),dim=1)
        return x, vq_loss, F.normalize(fe, dim=-1)

    def getFeatureCode(self, x):
        x = self.relu(self.bn1(self.conv1(x)))   # 128->64
        x = self.layer1(x)                       # 64
        x = self.layer2(x)                       # 32
        x = self.layer3(x)                       # 16
        # start_time = time.time()
        quantized_x, _ = self.quantizer_cb(x)
        x = quantized_x * 0.3 + x * 0.7
        # end_time=time.time()
        x = self.layer4(x)                       # 8
        x = self.avgpool(x)                      # 1×1
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)
        return x
