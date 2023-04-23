import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, reduce


class Layer(nn.Module):
  def __init__(self, d, nh):
    super().__init__()
    self.d, self.nh = d, nh
    self.wx, self.wo = nn.Linear(d, 3 * d), nn.Linear(d, d)
    self.ln = nn.LayerNorm(d)

  def forward(self, x, mask):
    proj = self.wx(self.ln(x))
    q, k, v = rearrange(proj, 'b l (n nh dh) -> n b nh l dh', n=3, nh=self.nh)
    qkT = einsum('bhic, bhjc -> bhij', q, k)
    if mask is not None:
      qkT = qkT + mask
    heads = F.softmax(qkT / self.d ** 0.5, -1) @ v
    attention = self.wo(rearrange(heads, 'b nh l dh -> b l (nh dh)'))
    return F.gelu(x + attention), mask


class LM(nn.Module):
  def __init__(self, d, nh, nl, l, v):
    super().__init__()
    self.emb = nn.Embedding(v, d)
    self.actor, self.critic = nn.Linear(d, v), nn.Linear(d, 1)
    self.layers = nn.ModuleList([Layer(d, nh) for _ in range(nl)])
    
    mask = torch.tril(torch.ones(l, l)) - 1
    mask[mask == -1] = float('-inf')
    self.mask = nn.Parameter(mask, requires_grad=False)

  def forward(self, x, actor=True):
    _, l = x.shape
    mask = self.mask[:l, :l] if actor else None
    x = self.emb(x)
    for layer in self.layers:
      x, mask = layer(x, mask)
    if not actor:
      return reduce(self.critic(x), 'b l 1 -> b', 'sum')
    return F.softmax(self.actor(x[:, -1]), -1)