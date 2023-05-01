import math
import logging
import json
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, Union


import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token

    def forward(self, x):
        if not self.training or self.prob == 0.:
            return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x



class ScaledAttention(nn.Module):
  'Cosine Scaled Attention'
  def __init__(self,
              dim, 
              num_of_head=8,
              bias=True,
              logit_scale_max=math.log(1. / 0.01),
              attn_drop=0.,
              proj_drop=0.
               ):
    super().__init__()
    assert dim % num_of_head == 0, 'dim should be divisible by num_heads'
    self.num_of_head = num_of_head
    self.head_dim = dim // num_of_head
    self.scale = self.head_dim ** -0.5 
    self.logit_scale_max = logit_scale_max 
    

    self.weight = nn.Parameter(torch.randn((dim * 3, dim)) * self.scale)
    if bias:
      self.bias = nn.Parameter(torch.zeros(dim * 3))
    else:
      self.bias = None

    self.attn_drop = nn.Dropout(attn_drop)
    self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_of_head, 1, 1))))
    self.out_proj = nn.Linear(dim, dim)
    self.out_drop = nn.Dropout(proj_drop)


  def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
    L, N, C = x.shape 
    q, k, v = F.linear(x, self.weight, self.bias).chunk(3, dim=-1)

    q = q.contiguous().view(1, N * self.num_of_head, -1).transpose(0, 1)
    k = k.contiguous().view(L, N * self.num_of_head, -1).transpose(0, 1)
    v = v.contiguous().view(L, N * self.num_of_head, -1).transpose(0, 1)

    key_norm = k.norm(p=2, dim=-1, keepdim=True)
    query_norm = q.norm(p=2, dim=-1, keepdim=True)

    cos_similarity = torch.matmul(q, k.transpose(-2, -1)) / (torch.matmul(query_norm, key_norm.transpose(-2, -1)))
    attn = cos_similarity / (self.scale ** 0.5)

    logit_scale = torch.clamp(self.logit_scale, max=self.logit_scale_max).exp()
    attn = attn.view(N, self.num_of_head, L, L) * logit_scale
    attn = attn.view(-1, L, L)

    if attn_mask is not None:
      if attn_mask.dtype == torch.bool:
          new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
          new_attn_mask.masked_fill_(attn_mask, float("-inf"))
          attn_mask = new_attn_mask
      attn += attn_mask

    attn = F.softmax(attn, dim=-1)
    attn = self.attn_drop(attn)
    x = torch.bmm(attn, v)

    x = x.transpose(0, 1).reshape(L, N, C)
    x = self.out_proj(x)
    x = self.out_drop(x)

    return x


x = torch.rand(1, 64, 64)
a = ScaledAttention(dim=64)
print(a(x))