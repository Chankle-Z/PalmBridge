import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes

class FeedForward(nn.Module):

    def __init__(self, dim, dim_for_mlp, dropout=0.1):
        super(FeedForward, self).__init__()

        self.dim = dim
        self.dim_for_mlp = dim_for_mlp
        self.dropout_rate = dropout

        self.net = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim_for_mlp),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.dim_for_mlp, self.dim),
            nn.Dropout(self.dropout_rate)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):

    def __init__(self, dim, heads, dim_for_head, dropout=0.1):
        super(Attention, self).__init__()

        self.dim = dim
        self.heads = heads
        self.dim_for_head = dim_for_head
        self.dropout_rate = dropout

        self.dim_for_head_inner = self.dim_for_head * self.heads

        self.project_out = not (self.heads == 1 and self.dim_for_head == self.dim)

        self.scale = self.dim_for_head ** -0.5

        self.norm = nn.LayerNorm(self.dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(self.dropout_rate)

        self.to_qkv = nn.Linear(self.dim, self.dim_for_head_inner * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(self.dim_for_head_inner, self.dim),
            nn.Dropout(self.dropout_rate)
        ) if self.project_out else nn.Identity()

    def forward(self, x):
        # torch.Size([batch_size, num_patches + 1, dim])
        x = self.norm(x)

        # torch.Size([batch_size, num_patches + 1, dim_for_head_inner * 3])
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        # torch.Size([batch_size, heads, num_patches + 1, dim_for_head_inner // heads = dim_for_head])
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # torch.Size([batch_size, heads, num_patches + 1, num_patches + 1])
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        # torch.Size([batch_size, heads, num_patches + 1, dim_for_head_inner // heads = dim_for_head])
        out = torch.matmul(attn, v)

        # torch.Size([batch_size, num_patches + 1, heads * dim_for_head_inner // heads])
        out = rearrange(out, 'b h n d -> b n (h d)')

        # torch.Size([batch_size, num_patches + 1, dim])
        return self.to_out(out)


# Transformer
class Transformer(nn.Module):

    def __init__(self, dim, depth, heads, dim_for_head, dim_for_mlp, dropout=0.1):
        super(Transformer, self).__init__()

        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dim_for_head = dim_for_head
        self.dim_for_mlp = dim_for_mlp
        self.dropout_rate = dropout

        self.norm = nn.LayerNorm(self.dim)
        self.layers = nn.ModuleList([])
        for _ in range(self.depth):
            self.layers.append(nn.ModuleList([
                Attention(heads=self.heads, dim=self.dim, dim_for_head=self.dim_for_head, dropout=self.dropout_rate),
                FeedForward(dim=self.dim, dim_for_mlp=self.dim_for_mlp, dropout=self.dropout_rate)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class ViT(nn.Module):

    def __init__(self, *, image_size, patch_size, channels, num_classes, depth, heads,
                 dim, dim_for_head, dim_for_mlp, pool='cls', dropout=0.1, emb_dropout=0.1):
        super(ViT, self).__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.channels = channels
        self.num_classes = num_classes
        self.depth = depth
        self.heads = heads
        self.dim = dim
        self.dim_for_head = dim_for_head
        self.dim_for_mlp = dim_for_mlp
        self.pool = pool
        self.dropout_rate = dropout
        self.emb_dropout_rate = emb_dropout

        self.image_height, self.image_width = pair(self.image_size)
        self.patch_height, self.patch_width = pair(self.patch_size)


        assert (self.image_height % self.patch_height == 0
                and self.image_width % self.patch_width == 0), ('Image dimensions must be '
                                                                'divisible by the patch size.')
        self.num_patches = (self.image_height // self.patch_height) * (self.image_width // self.patch_width)
        self.patch_dim = self.channels * self.patch_height * self. patch_width
        assert self.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_height, p2=self.patch_width),
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, self.dim),
            nn.LayerNorm(self.dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.dropout = nn.Dropout(self.emb_dropout_rate)

        self.transformer = Transformer(depth=self.depth, heads=self.heads, dim=self.dim,
                                       dim_for_head=self.dim_for_head, dim_for_mlp=self.dim_for_mlp,
                                       dropout=self.dropout_rate)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(self.dim, self.num_classes)

    def forward(self, feature_tensor):
        feature_tensor_for_vit = self.to_patch_embedding(feature_tensor)
        # torch.Size([batch_size, num_patches, dim])
        b, n, _ = feature_tensor_for_vit.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        feature_tensor_for_vit = torch.cat((cls_tokens, feature_tensor_for_vit), dim=1)
        feature_tensor_for_vit += self.pos_embedding[:, :(n + 1)]
        feature_tensor_for_vit = self.dropout(feature_tensor_for_vit)

        feature_tensor_for_vit = self.transformer(feature_tensor_for_vit)

        feature_tensor_for_vit = self.to_latent(feature_tensor_for_vit)

        return feature_tensor_for_vit
