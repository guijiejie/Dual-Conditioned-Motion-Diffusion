import torch
import torch.nn.functional as F
from torch import layer_norm, nn
import numpy as np
from typing import List, Tuple, Union

import math


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class StylizationBlock(nn.Module):

    def __init__(self, latent_dim, time_embed_dim, dropout):
        super().__init__()
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 2 * latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Linear(latent_dim, latent_dim)),
        )

    def forward(self, h, emb):
        """
        h: B, T, D
        emb: B, D
        """
        # B, 1, 2D
        emb_out = self.emb_layers(emb).unsqueeze(1)
        # scale: B, 1, D / shift: B, 1, D
        scale, shift = torch.chunk(emb_out, 2, dim=2)
        # B, T, D
        h = self.norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return h


class FFN(nn.Module):

    def __init__(self, latent_dim, ffn_dim, dropout, time_embed_dim):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, ffn_dim)
        self.linear2 = zero_module(nn.Linear(ffn_dim, latent_dim))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x, emb):
        """
            x: B, T, D (D=latent_dim)
        """
        y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        y = x + self.proj_out(y, emb)
        return y


class TemporalSelfAttention(nn.Module):

    def __init__(self, n_frames, latent_dim, num_head, dropout, time_embed_dim, output_attention = True):
        super().__init__()
        self.num_head = num_head
        self.output_attention = output_attention
        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim, bias=False)
        self.key = nn.Linear(latent_dim, latent_dim, bias=False)
        self.value = nn.Linear(latent_dim, latent_dim, bias=False)
        self.sigma_projection = nn.Linear(latent_dim, num_head, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)
        n_frames = n_frames
        self.distances = torch.zeros((n_frames, n_frames)).cuda(3)

        for i in range(n_frames):
            for j in range(n_frames):
                self.distances[i][j] = abs(i - j)

    def forward(self, x, emb):
        """
        x: B, T, D (D=latent_dim)
        """
        B, T, D = x.shape
        H = self.num_head

        ## series-association
        # B, T, 1, D
        query = self.query(self.norm(x)).unsqueeze(2)
        # B, 1, T, D
        key = self.key(self.norm(x)).unsqueeze(1)
        # B, T, H, D/H
        query = query.view(B, T, H, -1)
        key = key.view(B, T, H, -1)
        scale = 1. / math.sqrt(D/H)
        # B, H, T, T
        scores = torch.einsum('bnhd,bmhd->bhnm', query, key) / math.sqrt(D // H)
        attention = scale * scores
        # B, H, T, T
        series = self.dropout(F.softmax(attention, dim=-1))

        ## prior-association
        sigma = self.sigma_projection(x).view(B, T, H)  # B, T, H
        sigma = sigma.transpose(1, 2)  # B T H ->  B H T
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, T)  # B, H, T, T
        prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1) # B, H, T, T
        prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior ** 2 / 2 / (sigma ** 2)).cuda(3) # B, H, T, T

        # B, T, H, D/H
        value = self.value(self.norm(x)).view(B, T, H, -1)
        # B, T, D
        y = torch.einsum('bhnm,bmhd->bnhd', series, value).reshape(B, T, D)
        y = x + self.proj_out(y, emb)

        if self.output_attention:
            return y.contiguous(), series, prior, sigma
        else:
            return y.contiguous(), None

class TemporalDiffusionTransformerDecoderLayer(nn.Module):

    def __init__(self,
                 n_frames = 7,
                 latent_dim=16,
                 time_embed_dim=16,
                 ffn_dim=32,
                 num_head=4,
                 dropout=0.5
                 ):
        super().__init__()
        self.sa_block = TemporalSelfAttention(
            n_frames, latent_dim, num_head, dropout, time_embed_dim)
        self.ffn = FFN(latent_dim, ffn_dim, dropout, time_embed_dim)

    def forward(self, x, emb):
        x, series, prior, sigma = self.sa_block(x, emb)
        x = self.ffn(x, emb)
        return x, series, prior, sigma


class MotionTransformer(nn.Module):
    def __init__(self,
                 input_feats,
                 num_frames=7,
                 latent_dim=16,
                 ff_size=32,
                 num_layers=8,
                 num_heads=8,
                 dropout=0.2,
                 activation="gelu",
                 output_attention = True,
                 device: Union[str, torch.DeviceObjType] = 'cpu',
                 inject_condition: bool = False,
                 **kargs):
        super().__init__()


        self.input_feats = input_feats # 34
        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.output_attention = output_attention
        self.device = device
        self.time_embed_dim = latent_dim
        self.inject_condition = inject_condition

        self.build_model()

    def build_model(self):
        self.sequence_embedding = nn.Parameter(torch.randn(self.num_frames, self.latent_dim))

        # Input Embedding
        self.joint_embed = nn.Linear(self.input_feats, self.latent_dim)
        self.cond_embed = nn.Linear(256, self.time_embed_dim)

        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        self.temporal_decoder_blocks = nn.ModuleList()
        for i in range(self.num_layers):
            self.temporal_decoder_blocks.append(
                TemporalDiffusionTransformerDecoderLayer(
                    n_frames=self.num_frames,
                    latent_dim=self.latent_dim,
                    time_embed_dim=self.time_embed_dim,
                    ffn_dim=self.ff_size,
                    num_head=self.num_heads,
                    dropout=self.dropout,
                )
            )
        # Output Module
        self.out = zero_module(nn.Linear(self.latent_dim, self.input_feats))


    def forward(self, x, timesteps, condition_data:torch.Tensor=None):
        """
        x: B, T, D (D=C*V)
        """
        B, T = x.shape[0], x.shape[1]

        # B, latent_dim
        emb = self.time_embed(timestep_embedding(timesteps, self.latent_dim))

        # Add conditioning signal
        if self.inject_condition:
            condition_data = self.cond_embed(condition_data)
            emb = emb + condition_data

        # B, T, latent_dim
        h = self.joint_embed(x)
        h = h + self.sequence_embedding.unsqueeze(0)[:, :T, :]
        
        i = 0
        prelist = []
        series_list = []
        prior_list = []
        sigma_list = []
        for module in self.temporal_decoder_blocks:
            if i < (self.num_layers // 2):
                prelist.append(h)
                h, series, prior, sigmas = module(h, emb) # B, T, latent_dim
                series_list.append(series)
                prior_list.append(prior)
                sigma_list.append(sigmas)
            elif i >= (self.num_layers // 2):
                h, series, prior, sigmas = module(h, emb)
                h += prelist[-1]
                series_list.append(series)
                prior_list.append(prior)
                sigma_list.append(sigmas)
                prelist.pop()
            i += 1

        # B, T, C*V
        output = self.out(h).view(B, T, -1).contiguous()
        if self.output_attention:
            return output, series_list, prior_list, sigma_list
        return output
