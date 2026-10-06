# 加载模型
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from models.compnet import compnet

device = "cuda" if torch.cuda.is_available() else "cpu"

net = compnet(num_classes=230).to(device)   # num_classes 改成你训练时的
checkpoint = torch.load("./checkpoints/best_model.pth", map_location=device)
net.load_state_dict(checkpoint["state_dict"])  # 视保存方式可能需要改 key
net.eval()

@torch.no_grad()
def extract_features(data_loader, net, max_batches=10):
    """
    用训练好的 compnet 提取特征
    Args:
        data_loader: DataLoader
        net: 已经加载好权重的 compnet
        max_batches: 限制 batch 数，避免采样太多
    Returns:
        features: (N, 512) L2 归一化的特征向量
    """
    features = []

    for i, (x, _) in enumerate(data_loader):
        if i >= max_batches:
            break
        x = x.to(device)

        z = net.getFeatureCode(x)   # (B, 512)
        features.append(z.cpu())

    features = torch.cat(features, dim=0)
    print(f"[Feature Extract] collected {features.shape[0]} samples, dim={features.shape[1]}")
    return features
