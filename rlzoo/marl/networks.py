"""Joint actor architectures: all map agent inputs (B, n, d) -> action logits (B, n, A)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rlzoo.common.nets import get_activation, mlp


class PerAgentActor(nn.Module):
    """Separate (non-shared) policy per agent - used by HAPPO."""

    def __init__(self, n_agents, in_dim, n_actions, hidden_sizes, activation):
        super().__init__()
        self.nets = nn.ModuleList([mlp(in_dim, n_actions, hidden_sizes, activation) for _ in range(n_agents)])

    def forward(self, x):
        return torch.stack([net(x[..., i, :]) for i, net in enumerate(self.nets)], dim=-2)


class CommNetActor(nn.Module):
    """CommNet (Sukhbaatar et al., 2016): agents average their hidden states over K communication steps."""

    def __init__(self, in_dim, n_actions, hidden=64, comm_steps=2, activation="tanh"):
        super().__init__()
        self.act = get_activation(activation)()
        self.encoder = nn.Linear(in_dim, hidden)
        self.H = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(comm_steps)])
        self.C = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(comm_steps)])
        self.head = nn.Linear(hidden, n_actions)

    def forward(self, x):
        n = x.shape[-2]
        h0 = h = self.act(self.encoder(x))
        for H, C in zip(self.H, self.C):
            c = (h.sum(-2, keepdim=True) - h) / max(n - 1, 1)  # mean of the *other* agents' hidden states
            h = self.act(H(h) + C(c) + h0)
        return self.head(h)


class G2ANetActor(nn.Module):
    """G2ANet-style two-stage attention (Liu et al., 2020).

    Hard attention: a pairwise gate decides whether agent j is relevant to agent i (straight-through
    threshold, deterministic so PPO ratios stay exact). Soft attention: scaled dot-product weights over the
    gated agents. The policy sees [own embedding, aggregated message].
    """

    def __init__(self, in_dim, n_actions, hidden=64, activation="relu"):
        super().__init__()
        self.hidden = hidden
        self.encoder = nn.Sequential(nn.Linear(in_dim, hidden), get_activation(activation)())
        self.hard = nn.Sequential(nn.Linear(2 * hidden, hidden), get_activation(activation)(), nn.Linear(hidden, 1))
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.k = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Sequential(nn.Linear(hidden, hidden), get_activation(activation)())
        self.head = mlp(2 * hidden, n_actions, [hidden], activation)

    def gates(self, h):
        n = h.shape[-2]
        hi = h.unsqueeze(-2).expand(*h.shape[:-1], n, self.hidden)
        hj = h.unsqueeze(-3).expand(*h.shape[:-2], n, n, self.hidden)
        p = torch.sigmoid(self.hard(torch.cat([hi, hj], -1)).squeeze(-1))  # (B, n, n)
        return (p > 0.5).float() + p - p.detach()

    def forward(self, x):
        n = x.shape[-2]
        h = self.encoder(x)
        gate = self.gates(h)
        not_self = 1.0 - torch.eye(n, device=x.device)
        scores = (self.q(h) @ self.k(h).transpose(-1, -2)) / math.sqrt(self.hidden)
        scores = scores.masked_fill(not_self == 0, -1e9)
        w = F.softmax(scores, -1) * gate * not_self
        msg = w @ self.v(h)
        return self.head(torch.cat([h, msg], -1))


class MATDecodeBlock(nn.Module):
    """MAT decoder block: masked self-attention over action tokens, then cross-attention whose *query* is the
    encoder representation of each agent (so position i keeps agent i's observation) with keys/values from the
    action tokens of agents < i."""

    def __init__(self, embed, heads):
        super().__init__()
        self.attn1 = nn.MultiheadAttention(embed, heads, batch_first=True)
        self.attn2 = nn.MultiheadAttention(embed, heads, batch_first=True)
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(embed), nn.LayerNorm(embed), nn.LayerNorm(embed)
        self.mlp = nn.Sequential(nn.Linear(embed, 2 * embed), nn.GELU(), nn.Linear(2 * embed, embed))

    def forward(self, x, rep, mask):
        x = self.ln1(x + self.attn1(x, x, x, attn_mask=mask, need_weights=False)[0])
        x = self.ln2(rep + self.attn2(rep, x, x, attn_mask=mask, need_weights=False)[0])
        return self.ln3(x + self.mlp(x))


class MATNetwork(nn.Module):
    """Multi-Agent Transformer (Wen et al., 2022): encoder over observations (with value head) and an
    autoregressive decoder that emits agent i's action conditioned on actions of agents < i."""

    def __init__(self, in_dim, n_actions, n_agents, embed=64, heads=2, blocks=1):
        super().__init__()
        self.A, self.n = n_actions, n_agents
        self.obs_enc = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, embed), nn.GELU())
        self.enc = nn.ModuleList([nn.TransformerEncoderLayer(embed, heads, 2 * embed, dropout=0.0, batch_first=True,
                                                             norm_first=True) for _ in range(blocks)])
        self.enc_ln = nn.LayerNorm(embed)
        self.value_head = nn.Sequential(nn.Linear(embed, embed), nn.GELU(), nn.Linear(embed, 1))
        self.act_enc = nn.Sequential(nn.Linear(n_actions + 1, embed), nn.GELU())
        self.dec = nn.ModuleList([MATDecodeBlock(embed, heads) for _ in range(blocks)])
        self.dec_ln = nn.LayerNorm(embed)
        self.policy_head = nn.Sequential(nn.Linear(embed, embed), nn.GELU(), nn.Linear(embed, n_actions))
        self.register_buffer("causal", torch.triu(torch.ones(n_agents, n_agents, dtype=torch.bool), diagonal=1))

    def encode(self, x):
        h = self.obs_enc(x)
        for blk in self.enc:
            h = blk(h)
        h = self.enc_ln(h)
        return h, self.value_head(h).squeeze(-1)

    def decode(self, rep, shifted):
        k = shifted.shape[1]
        y = self.act_enc(shifted)
        for blk in self.dec:
            y = blk(y, rep[:, :k], self.causal[:k, :k])
        return self.policy_head(self.dec_ln(y))

    def shift(self, actions_onehot):
        """Teacher-forcing decoder inputs: [start, a_0, ..., a_{n-2}] with a leading start-flag channel."""
        B = actions_onehot.shape[0]
        start = torch.zeros(B, 1, self.A + 1, device=actions_onehot.device)
        start[..., 0] = 1
        prev = F.pad(actions_onehot[:, :-1], (1, 0))
        return torch.cat([start, prev], 1)
