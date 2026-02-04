
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def drop_path(x, drop_prob: float = 0., training: bool = False):

    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    def __init__(self, embed_dim=768, norm_layer=None):
        super().__init__()
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        return x


class MultiTimeSurvivalHead(nn.Module):
    """多时间点生存预测头"""

    def __init__(self, input_dim, hidden_dim=256, output_dim=3, dropout=0.3):
        super().__init__()
        self.output_dim = output_dim  # 3: 1年, 3年, 5年

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )


        self.time_bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        # x: [B, feature_dim]
        output = self.mlp(x)
        output = output + self.time_bias
        return output  # [B, 3]

class EmbedReduction(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=1280, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,   #
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):

        B, N, C = x.shape


        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]


        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)


        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ClinicalContextEncoder(nn.Module):

    def __init__(self, clinical_dim, num_context_tokens=32, context_dim=256, num_cross_attn_layers=2):
        super().__init__()
        self.num_context_tokens = num_context_tokens
        self.context_dim = context_dim


        self.context_queries = nn.Parameter(torch.randn(1, num_context_tokens, context_dim))


        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, context_dim)
        )


        self.cross_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=context_dim,
                nhead=8,
                dim_feedforward=context_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            for _ in range(num_cross_attn_layers)
        ])


        self.cross_attn = nn.MultiheadAttention(
            embed_dim=context_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )


        self.context_self_attn = nn.TransformerEncoderLayer(
            d_model=context_dim,
            nhead=8,
            dim_feedforward=context_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )

    def forward(self, clinical_features):

        B = clinical_features.shape[0]


        clinical_encoded = self.clinical_encoder(clinical_features).unsqueeze(1)  # [B, 1, context_dim]


        context_queries = self.context_queries.expand(B, -1, -1)  # [B, num_context_tokens, context_dim]


        context_tokens, _ = self.cross_attn(
            query=context_queries,
            key=clinical_encoded,
            value=clinical_encoded,
            need_weights=False
        )


        for layer in self.cross_attn_layers:

            context_tokens = layer(context_tokens + clinical_encoded)


        context_tokens = self.context_self_attn(context_tokens)

        return context_tokens



class ContextGuidedAttention(nn.Module):


    def __init__(self, dim, num_heads, context_dim=None, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5


        self.context_to_q = nn.Linear(dim, dim)
        self.context_to_kv = nn.Linear(dim, dim * 2)


        self.wsi_to_q = nn.Linear(dim, dim)
        self.wsi_to_kv = nn.Linear(dim, dim * 2)


        self.fusion_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )


        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, wsi_features, context_tokens=None):

        B, N, D = wsi_features.shape


        if context_tokens is None:

            q = self.wsi_to_q(wsi_features).reshape(B, N, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)
            k = self.wsi_to_kv(wsi_features)[:, :, :D].reshape(B, N, self.num_heads, D // self.num_heads).permute(0, 2,
                                                                                                                  1, 3)
            v = self.wsi_to_kv(wsi_features)[:, :, D:].reshape(B, N, self.num_heads, D // self.num_heads).permute(0, 2,
                                                                                                                  1, 3)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)

            x = (attn @ v).transpose(1, 2).reshape(B, N, D)
            x = self.proj(x)
            x = self.proj_drop(x)

            return x, attn

        M = context_tokens.shape[1]
        H = self.num_heads


        context_q = self.context_to_q(context_tokens).reshape(B, M, H, D // H).permute(0, 2, 1, 3)
        wsi_kv = self.wsi_to_kv(wsi_features).reshape(B, N, 2, H, D // H).permute(2, 0, 3, 1, 4)
        wsi_k, wsi_v = wsi_kv[0], wsi_kv[1]


        context_attn = (context_q @ wsi_k.transpose(-2, -1)) * self.scale
        context_attn = F.softmax(context_attn, dim=-1)  # [B, H, M, N]


        context_aware_features = (context_attn @ wsi_v).permute(0, 2, 1, 3).reshape(B, M, D)


        wsi_q = self.wsi_to_q(wsi_features).reshape(B, N, H, D // H).permute(0, 2, 1, 3)
        context_kv = self.context_to_kv(context_aware_features).reshape(B, M, 2, H, D // H).permute(2, 0, 3, 1, 4)
        context_k, context_v = context_kv[0], context_kv[1]


        wsi_attn = (wsi_q @ context_k.transpose(-2, -1)) * self.scale
        wsi_attn = F.softmax(wsi_attn, dim=-1)  # [B, H, N, M]


        context_injected = (wsi_attn @ context_v).permute(0, 2, 1, 3).reshape(B, N, D)


        fusion_weight = self.fusion_gate(torch.cat([wsi_features, context_injected], dim=-1))
        output = fusion_weight * wsi_features + (1 - fusion_weight) * context_injected

        output = self.proj(output)
        output = self.proj_drop(output)

        return output, wsi_attn



class KnowledgeGuidedGating(nn.Module):
    def __init__(self, dim, num_heads=8, clinical_dim=0, hypoxia_pathways=0, use_context_tokens=False,use_gating=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_context_tokens = use_context_tokens
        self.use_gating = use_gating

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)


        if use_gating:
            if clinical_dim > 0 and not use_context_tokens:
                self.clinical_bias = nn.Sequential(
                    nn.Linear(clinical_dim, num_heads),
                    nn.Tanh()
                )
            else:
                self.clinical_bias = None

            if hypoxia_pathways > 0:
                self.hypoxia_bias = nn.Sequential(
                    nn.Linear(hypoxia_pathways, num_heads),
                    nn.Tanh()
                )
            else:
                self.hypoxia_bias = None

    def forward(self, x, clinical_features=None, hypoxia_features=None, wsi_features=None, context_tokens=None):
        B, N, D = x.shape
        H = self.num_heads

        q = self.to_q(x).reshape(B, N, H, self.head_dim).permute(0, 2, 1, 3)
        k = self.to_k(x).reshape(B, N, H, self.head_dim).permute(0, 2, 1, 3)
        v = self.to_v(x).reshape(B, N, H, self.head_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale


        if self.use_gating:
            bias = torch.zeros(B, H, N, N, device=x.device)


            if self.clinical_bias is not None and clinical_features is not None and not self.use_context_tokens:
                clinical_bias = self.clinical_bias(clinical_features)  # [B, H]
                clinical_bias = clinical_bias.unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]

                for b in range(B):
                    for h in range(H):
                        bias[b, h].fill_diagonal_(clinical_bias[b, h, 0, 0])


            if self.hypoxia_bias is not None and hypoxia_features is not None:
                hypoxia_bias = self.hypoxia_bias(hypoxia_features)  # [B, H]
                hypoxia_bias = hypoxia_bias.unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]

                bias = bias + hypoxia_bias.expand(-1, -1, N, N)

            attn = attn + bias

        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)

        return out, attn

    def compute_necrosis_weight(self, wsi_features, hypoxia_features):
        B = wsi_features.shape[0]
        H = self.num_heads

        hypoxia_weight = hypoxia_features.mean(dim=-1, keepdim=True)  # [B, 1]
        hypoxia_weight = hypoxia_weight.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1]
        hypoxia_weight = hypoxia_weight.expand(-1, H, -1, -1)  # [B, H, 1, 1]

        return hypoxia_weight



class KnowledgeGuided_Fusion(nn.Module):
    def __init__(self, dim=256, num_heads=16, attn_drop_ratio=0., proj_drop_ratio=0.,
                 clinical_dim=0, hypoxia_pathways=10, use_context_tokens=False, chunk_size=64, use_gating=True):
        super(KnowledgeGuided_Fusion, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.use_context_tokens = use_context_tokens
        self.chunk_size = chunk_size
        self.use_gating = use_gating

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.scale = (dim // num_heads) ** -0.5

        if clinical_dim > 0:
            self.clinical_attention_bias = nn.Sequential(
                nn.Linear(clinical_dim, num_heads * dim // num_heads),
                nn.GELU(),
                nn.Linear(num_heads * dim // num_heads, num_heads)
            )
        else:
            self.clinical_attention_bias = None

        if hypoxia_pathways > 0:
            self.hypoxia_attention_bias = nn.Sequential(
                nn.Linear(hypoxia_pathways, num_heads * dim // num_heads),
                nn.GELU(),
                nn.Linear(num_heads * dim // num_heads, num_heads)
            )
        else:
            self.hypoxia_attention_bias = None

    def forward(self, wsi_features, gene_features, clinical_features=None, hypoxia_features=None,
                context_tokens=None):
        B, N, D = gene_features.shape
        M = wsi_features.shape[1]
        H = self.num_heads

        q = self.q(gene_features).reshape(B, N, H, D // H).permute(0, 2, 1, 3)
        k = self.k(wsi_features).reshape(B, M, H, D // H).permute(0, 2, 1, 3)
        v = self.v(wsi_features).reshape(B, M, H, D // H).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale


        if clinical_features is not None and self.clinical_attention_bias is not None:
            clinical_attn_bias = self.clinical_attention_bias(clinical_features)  # [B, H]
            clinical_attn_bias = clinical_attn_bias.unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]

            attn = attn + clinical_attn_bias.expand(-1, -1, N, M)

        if hypoxia_features is not None and self.hypoxia_attention_bias is not None:
            hypoxia_attn_bias = self.hypoxia_attention_bias(hypoxia_features)  # [B, H]
            hypoxia_attn_bias = hypoxia_attn_bias.unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]

            attn = attn + hypoxia_attn_bias.expand(-1, -1, N, M)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)

        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ContextAwareWSIBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_context=True):
        super().__init__()
        self.use_context = use_context


        self.norm1 = norm_layer(dim)


        if use_context:
            self.attn = ContextGuidedAttention(
                dim=dim,
                num_heads=num_heads,
                dropout=attn_drop_ratio
            )
        else:
            self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                 qk_scale=qk_scale, attn_drop_ratio=attn_drop_ratio,
                                 proj_drop_ratio=drop_ratio)


        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()

        # MLP
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                      act_layer=act_layer, drop=drop_ratio)

    def forward(self, x, context_tokens=None):

        if self.use_context:

            attn_out, attn_weights = self.attn(self.norm1(x), context_tokens)
        else:

            attn_out = self.attn(self.norm1(x))
            attn_weights = None

        x = x + self.drop_path(attn_out)


        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, attn_weights


class VisionTransformer(nn.Module):
    def __init__(self, wsi_patches=500, gene_patches=656, embed_wsi_dim=768, embed_gene_dim=256, num_classes=3,
                 depth_gene=3, depth_wsi=12, depth_fusion=4, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, clinical_dim=0, hypoxia_pathways=0,
                 num_context_tokens=32, use_context_tokens=True, gene_input_dim=64, chunk_size=64,use_gating=True):

        super(VisionTransformer, self).__init__()
        self.wsi_patches = wsi_patches
        self.gene_patches = gene_patches
        self.embed_wsi_dim = embed_wsi_dim
        self.embed_gene_dim = embed_gene_dim
        self.num_classes = num_classes
        self.use_context_tokens = use_context_tokens
        self.gene_input_dim = gene_input_dim
        self.use_gating = use_gating

        self.attention_weights = []
        self.register_buffer('_dummy', torch.zeros(1))
        self.clinical_dim = clinical_dim
        self.hypoxia_pathways = hypoxia_pathways
        self.num_context_tokens = num_context_tokens if clinical_dim > 0 and use_context_tokens else 0

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(embed_dim=embed_wsi_dim)
        self.gene_embed = EmbedReduction(in_features=gene_input_dim, hidden_features=128,
                                         out_features=self.embed_gene_dim,
                                         act_layer=act_layer, drop=drop_ratio)


        if clinical_dim > 0 and use_context_tokens:
            self.clinical_context_encoder = ClinicalContextEncoder(
                clinical_dim=clinical_dim,
                num_context_tokens=num_context_tokens,
                context_dim=embed_wsi_dim,
                num_cross_attn_layers=2
            )
        else:
            self.clinical_context_encoder = None


        if use_context_tokens and clinical_dim > 0:

            total_wsi_tokens = wsi_patches + num_context_tokens
            self.pos_wsi_embed = nn.Parameter(torch.zeros(1, total_wsi_tokens, embed_wsi_dim))


            self.context_pos_embed = nn.Parameter(torch.zeros(1, num_context_tokens, embed_wsi_dim))
        else:

            self.pos_wsi_embed = nn.Parameter(torch.zeros(1, wsi_patches, embed_wsi_dim))
            self.context_pos_embed = None


        self.pos_gene_embed = nn.Parameter(torch.zeros(1, self.gene_patches, self.embed_gene_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)


        if clinical_dim > 0 and not use_context_tokens:
            self.clinical_encoder = nn.Sequential(
                nn.Linear(clinical_dim, 64),
                nn.ReLU(),
                nn.Dropout(drop_ratio),
                nn.Linear(64, 32)
            )
        else:
            self.clinical_encoder = None


        if hypoxia_pathways > 0:
            self.hypoxia_encoder = nn.Sequential(
                nn.Linear(hypoxia_pathways, 32),
                nn.ReLU(),
                nn.Dropout(drop_ratio),
                nn.Linear(32, 16)
            )
        else:
            self.hypoxia_encoder = None

        if use_context_tokens and clinical_dim > 0:

            self.context_pooling = nn.Sequential(
                nn.Linear(num_context_tokens * embed_wsi_dim, 256),
                nn.GELU(),
                nn.Dropout(drop_ratio),
                nn.Linear(256, 128)
            )
            print(f"[DEBUG] 初始化 context_pooling 层: {num_context_tokens * embed_wsi_dim} -> 128")
        else:
            self.context_pooling = None

        dpr_wsi = [x.item() for x in torch.linspace(0, drop_path_ratio, depth_wsi)]
        self.blocks_wsi = nn.ModuleList([
            ContextAwareWSIBlock(
                dim=self.embed_wsi_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_ratio=drop_ratio,
                attn_drop_ratio=attn_drop_ratio,
                drop_path_ratio=dpr_wsi[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_context=(use_context_tokens and clinical_dim > 0)
            )
            for i in range(depth_wsi)
        ])


        dpr_gene = [x.item() for x in torch.linspace(0, drop_path_ratio, depth_gene)]
        self.blocks_gene = nn.Sequential(*[
            Block(dim=self.embed_gene_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr_gene[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth_gene)
        ])

        self.norm_wsi = norm_layer(self.embed_wsi_dim)
        self.norm_gene = norm_layer(self.embed_gene_dim)

        self.wsi_embed_reduction = EmbedReduction(in_features=self.embed_wsi_dim,
                                                  hidden_features=640,
                                                  out_features=self.embed_gene_dim,
                                                  act_layer=act_layer, drop=drop_ratio)


        self.gene_guided_wsi_fusion = KnowledgeGuided_Fusion(
            dim=self.embed_gene_dim,
            num_heads=num_heads,
            clinical_dim=32 if (clinical_dim > 0 and not use_context_tokens) else 0,
            hypoxia_pathways=16 if hypoxia_pathways > 0 else 0,
            attn_drop_ratio=attn_drop_ratio,
            proj_drop_ratio=drop_ratio,
            use_context_tokens=use_context_tokens and clinical_dim > 0,
            chunk_size=chunk_size,
            use_gating=use_gating
        )

        self.gap_fusion = nn.AdaptiveAvgPool2d((self.gene_patches, 1))
        self.gap_gene = nn.AdaptiveAvgPool2d((self.gene_patches, 1))


        head_input_dim = 0
        head_input_dim += self.gene_patches * 2

        if clinical_dim > 0 and not use_context_tokens:
            head_input_dim += 32

        if use_context_tokens and clinical_dim > 0:
            head_input_dim += 128

        if hypoxia_pathways > 0:
            head_input_dim += 16

        print(f"\n[DEBUG] 分类头维度: {head_input_dim} -> {num_classes}")

        if num_classes == 1:

            self.head = nn.Linear(head_input_dim, num_classes) if num_classes > 0 else nn.Identity()
            self.survival_head = None
        else:

            self.head = None
            self.survival_head = nn.Sequential(
                nn.Linear(head_input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )
            print(f"[INFO] 创建多时间点生存预测头: {head_input_dim} -> {num_classes}")

        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_gene_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        nn.init.trunc_normal_(self.pos_wsi_embed, std=0.02)
        nn.init.trunc_normal_(self.pos_gene_embed, std=0.02)
        if self.context_pos_embed is not None:
            nn.init.trunc_normal_(self.context_pos_embed, std=0.02)

        self.apply(_init_vit_weights)

    def forward_features_wsi(self, x, context_tokens=None):


        if context_tokens is not None:
            pass


        if x.dim() == 2:
            x = x.unsqueeze(0)  # [N, D] -> [1, N, D]


        x = self.patch_embed(x)  # [B, N, D]



        if context_tokens is not None:

            x = torch.cat([context_tokens, x], dim=1)


        if x.shape[1] != self.pos_wsi_embed.shape[1]:

            if x.shape[1] < self.pos_wsi_embed.shape[1]:

                pos_embed = self.pos_wsi_embed[:, :x.shape[1], :]
                x = self.pos_drop(x + pos_embed)

            else:

                if x.shape[1] > self.pos_wsi_embed.shape[1]:
                    x = x[:, :self.pos_wsi_embed.shape[1], :]
                x = self.pos_drop(x + self.pos_wsi_embed)

        else:
            x = self.pos_drop(x + self.pos_wsi_embed)



        all_attn_weights = []
        for i, block in enumerate(self.blocks_wsi):

            x, attn_weights = block(
                x,
                context_tokens
            )
            if attn_weights is not None:
                all_attn_weights.append(attn_weights)


        x = self.norm_wsi(x)



        context_tokens_out = None
        if context_tokens is not None:

            context_tokens_out = x[:, :context_tokens.shape[1], :]
            x = x[:, context_tokens.shape[1]:, :]


        return x, context_tokens_out, all_attn_weights

    def forward_features_gene(self, x):


        if x.dim() == 2:
            x = x.unsqueeze(0)  # [N, D] -> [1, N, D]


        x = self.pos_drop(x + self.pos_gene_embed)


        x = self.blocks_gene(x)


        x = self.norm_gene(x)

        return x

    def forward(self, x_wsi, x_gene, clinical_data=None, hypoxia_data=None):

        gene2wsi_feature = None
        pred_head = None
        fusion_attn = None
        attention_weights_gpu = []


        if x_wsi.dim() == 2:
            x_wsi = x_wsi.unsqueeze(0)
        if x_gene.dim() == 2:
            x_gene = x_gene.unsqueeze(0)

        B = x_wsi.shape[0]

        print(f"\n[FORWARD] Batch size: {B}")
        print(f"[FORWARD] num_classes: {self.num_classes}")
        print(f"[FORWARD] 使用survival_head: {self.num_classes != 1}")


        clinical_features_old = None
        context_tokens = None

        try:
            if self.clinical_dim > 0 and clinical_data is not None:
                if clinical_data.dim() == 1:
                    clinical_data = clinical_data.unsqueeze(0)

                if self.use_context_tokens:
                    context_tokens = self.clinical_context_encoder(clinical_data)
                else:
                    clinical_features_old = self.clinical_encoder(clinical_data)
        except Exception as e:
            print(f"[ERROR] 临床数据处理失败: {e}")


        hypoxia_features = None
        try:
            if self.hypoxia_pathways > 0 and hypoxia_data is not None:
                if hypoxia_data.dim() == 1:
                    hypoxia_data = hypoxia_data.unsqueeze(0)

                if hasattr(self, 'hypoxia_encoder') and self.hypoxia_encoder is not None:
                    hypoxia_features = self.hypoxia_encoder(hypoxia_data)
        except Exception as e:
            print(f"[ERROR] 缺氧数据处理失败: {e}")


        wsi_features = None
        context_tokens_out = None
        wsi_attn_weights = []

        try:
            wsi_features, context_tokens_out, wsi_attn_weights = self.forward_features_wsi(x_wsi, context_tokens)


            if wsi_attn_weights:
                for layer_attn in wsi_attn_weights:
                    if layer_attn is not None:
                        attention_weights_gpu.append(layer_attn.detach())
        except Exception as e:
            print(f"[ERROR] WSI特征提取失败: {e}")

            wsi_features = torch.randn(B, self.wsi_patches, self.embed_wsi_dim).to(x_wsi.device)

        if context_tokens_out is None and context_tokens is not None:
            context_tokens_out = context_tokens


        gene_features = None
        gene_features_reduction = None
        try:
            gene_features_reduction = self.gene_embed(x_gene)
            gene_features = self.forward_features_gene(gene_features_reduction)
        except Exception as e:
            print(f"[ERROR] 基因特征提取失败: {e}")
            # 创建默认的基因特征
            gene_features = torch.randn(B, self.gene_patches, self.embed_gene_dim).to(x_gene.device)
            gene_features_reduction = gene_features


        try:
            if wsi_features is not None and gene_features_reduction is not None:
                wsi_features_reduction = self.wsi_embed_reduction(wsi_features)
                gene2wsi_feature = gene_features_reduction @ wsi_features_reduction.transpose(-2, -1)
            else:

                gene2wsi_feature = torch.randn(B, self.gene_patches, self.wsi_patches).to(x_wsi.device)
        except Exception as e:
            print(f"[ERROR] 计算gene2wsi_feature失败: {e}")
            gene2wsi_feature = torch.randn(B, self.gene_patches, self.wsi_patches).to(x_wsi.device)


        fusion_context_tokens = context_tokens_out if context_tokens_out is not None else context_tokens

        try:
            if wsi_features is not None and gene_features is not None:
                wsi_features_reduction = self.wsi_embed_reduction(wsi_features)
                x, fusion_attn = self.gene_guided_wsi_fusion(
                    wsi_features_reduction,
                    gene_features,
                    clinical_features_old,
                    hypoxia_features,
                    fusion_context_tokens
                )

                x = self.gap_fusion(x)
                gene_features_gap = self.gap_gene(gene_features)

                fused_features = torch.cat([gene_features_gap, x], dim=1)
                fused_features = fused_features.squeeze(2)

                if clinical_features_old is not None:
                    fused_features = torch.cat([fused_features, clinical_features_old], dim=1)

                if context_tokens_out is not None and self.context_pooling is not None:
                    context_pooled = context_tokens_out.reshape(B, -1)
                    context_features = self.context_pooling(context_pooled)
                    fused_features = torch.cat([fused_features, context_features], dim=1)

                if hypoxia_features is not None:
                    fused_features = torch.cat([fused_features, hypoxia_features], dim=1)

                print(f"[DEBUG] 融合特征维度: {fused_features.shape}")
            else:

                fused_features = torch.randn(B, 1456).to(x_wsi.device)
        except Exception as e:
            print(f"[ERROR] 特征融合失败: {e}")
            fused_features = torch.randn(B, 1456).to(x_wsi.device)


        try:
            if self.num_classes == 1:

                if self.head is not None:
                    pred_head = self.head(fused_features)
                else:
                    print(f"[INFO] 创建新的分类头: {fused_features.shape[1]} -> {self.num_classes}")
                    self.head = nn.Linear(fused_features.shape[1], self.num_classes).to(fused_features.device)
                    pred_head = self.head(fused_features)
            else:

                if self.survival_head is not None:
                    pred_head = self.survival_head(fused_features)
                else:
                    print(f"[WARNING] survival_head不存在，使用线性层")
                    pred_head = nn.Linear(fused_features.shape[1], self.num_classes).to(fused_features.device)(
                        fused_features)
        except Exception as e:
            print(f"[ERROR] 生成预测失败: {e}")

            pred_head = torch.zeros(B, self.num_classes).to(x_wsi.device)


        if pred_head is None:
            print(f"[ERROR] pred_head为None，使用零张量")
            pred_head = torch.zeros(B, self.num_classes).to(x_wsi.device)

        if gene2wsi_feature is None:
            print(f"[ERROR] gene2wsi_feature为None，使用随机张量")
            gene2wsi_feature = torch.randn(B, self.gene_patches, self.wsi_patches).to(x_wsi.device)

        if fusion_attn is None:
            print(f"[INFO] fusion_attn为None，使用空张量")
            fusion_attn = torch.zeros(B, 1, 1, 1).to(x_wsi.device)


        return (gene2wsi_feature,
                pred_head,
                fusion_attn,
                attention_weights_gpu)


def _init_vit_weights(m):

    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def my_model(num_classes: int = 3, has_logits: bool = True, wsi_block=12, gene_block=3,
             dpr=0.1, clinical_dim=0, hypoxia_pathways=0, use_context_tokens=False,
             embed_wsi_dim=768, embed_gene_dim=256, wsi_patches=500, gene_patches=656,
             gene_input_dim=64,use_gating=True):
    print(f"[DEBUG] 创建多时间点生存预测模型")
    print(f"  - 输出维度: {num_classes} (1年、3年、5年风险评分)")
    model = VisionTransformer(
        wsi_patches=wsi_patches,
        gene_patches=gene_patches,
        embed_wsi_dim=embed_wsi_dim,
        embed_gene_dim=embed_gene_dim,
        depth_gene=gene_block,
        depth_wsi=wsi_block,
        num_heads=16,
        representation_size=2048 if has_logits else None,
        drop_path_ratio=dpr,
        drop_ratio=dpr,
        attn_drop_ratio=dpr,
        num_classes=num_classes,
        clinical_dim=clinical_dim,
        hypoxia_pathways=hypoxia_pathways,
        use_context_tokens=use_context_tokens,
        num_context_tokens=32,
        gene_input_dim=gene_input_dim,
        chunk_size=64,
        use_gating=use_gating
    )
    return model