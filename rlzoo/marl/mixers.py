"""Value-factorisation mixing networks. All take agent Qs (B, n) and state (B, s) and return Q_tot (B,)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VDNMixer(nn.Module):
    def forward(self, qs, state, **_):
        return qs.sum(-1)


class QMixer(nn.Module):
    """Monotonic hypernetwork mixer (Rashid et al., 2018)."""

    def __init__(self, n_agents, state_dim, embed_dim=32, hypernet_embed=64):
        super().__init__()
        self.n, self.embed = n_agents, embed_dim
        self.hyper_w1 = nn.Sequential(nn.Linear(state_dim, hypernet_embed), nn.ReLU(),
                                      nn.Linear(hypernet_embed, embed_dim * n_agents))
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.hyper_w2 = nn.Sequential(nn.Linear(state_dim, hypernet_embed), nn.ReLU(),
                                      nn.Linear(hypernet_embed, embed_dim))
        self.hyper_b2 = nn.Sequential(nn.Linear(state_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1))

    def forward(self, qs, state, **_):
        B = qs.shape[0]
        w1 = self.hyper_w1(state).abs().view(B, self.n, self.embed)
        b1 = self.hyper_b1(state).view(B, 1, self.embed)
        hidden = F.elu(torch.bmm(qs.view(B, 1, self.n), w1) + b1)
        w2 = self.hyper_w2(state).abs().view(B, self.embed, 1)
        b2 = self.hyper_b2(state).view(B, 1, 1)
        return (torch.bmm(hidden, w2) + b2).view(B)


class QattenMixer(nn.Module):
    """Qatten (Yang et al., 2020): multi-head attention over the agents' Q-values.

    Q_tot = sum_h lambda_h(s) sum_i alpha_{h,i} Q_i + V(s), with alpha_h = softmax_i(q_h(s) . k_h(o_i)) and
    lambda_h(s) >= 0 (abs of a hypernetwork output), so Q_tot stays monotonic in every Q_i.
    """

    def __init__(self, n_agents, state_dim, obs_dim, n_heads=4, embed_dim=32, hypernet_embed=64):
        super().__init__()
        self.n, self.H, self.embed = n_agents, n_heads, embed_dim
        self.query = nn.ModuleList([nn.Sequential(nn.Linear(state_dim, hypernet_embed), nn.ReLU(),
                                                  nn.Linear(hypernet_embed, embed_dim)) for _ in range(n_heads)])
        self.key = nn.ModuleList([nn.Linear(obs_dim, embed_dim) for _ in range(n_heads)])
        self.lam = nn.Sequential(nn.Linear(state_dim, hypernet_embed), nn.ReLU(), nn.Linear(hypernet_embed, n_heads))
        self.v = nn.Sequential(nn.Linear(state_dim, hypernet_embed), nn.ReLU(), nn.Linear(hypernet_embed, 1))

    def forward(self, qs, state, obs=None, **_):
        B = qs.shape[0]
        lam = self.lam(state).abs()  # (B, H)
        out = 0.0
        for h in range(self.H):
            q = self.query[h](state).view(B, 1, self.embed)
            k = self.key[h](obs)  # (B, n, embed)
            alpha = F.softmax((k * q).sum(-1) / self.embed ** 0.5, -1)  # (B, n)
            out = out + lam[:, h] * (alpha * qs).sum(-1)
        return out + self.v(state).squeeze(-1)


class QPLEXMixer(nn.Module):
    """Duplex dueling mixer (Wang et al., 2021).

    Q_tot = sum_i Q_i' + sum_i (lambda_i(s, a) - 1) * A_i', with Q_i' = w_i(s) Q_i + b_i(s),
    A_i' = Q_i' - V_i' and lambda_i > 0 from multi-head attention over state and joint action.
    """

    def __init__(self, n_agents, state_dim, n_actions, hypernet_embed=64, n_heads=4):
        super().__init__()
        self.n, self.H = n_agents, n_heads
        self.hyper_w = nn.Sequential(nn.Linear(state_dim, hypernet_embed), nn.ReLU(),
                                     nn.Linear(hypernet_embed, n_agents))
        self.hyper_b = nn.Sequential(nn.Linear(state_dim, hypernet_embed), nn.ReLU(),
                                     nn.Linear(hypernet_embed, n_agents))
        self.key = nn.Sequential(nn.Linear(state_dim, hypernet_embed), nn.ReLU(), nn.Linear(hypernet_embed, n_heads))
        self.agents = nn.Sequential(nn.Linear(state_dim, hypernet_embed), nn.ReLU(),
                                    nn.Linear(hypernet_embed, n_heads * n_agents))
        self.action = nn.Sequential(nn.Linear(state_dim + n_agents * n_actions, hypernet_embed), nn.ReLU(),
                                    nn.Linear(hypernet_embed, n_heads * n_agents))

    def forward(self, qs, state, actions_onehot=None, max_qs=None, **_):
        B = qs.shape[0]
        w = self.hyper_w(state).abs() + 1e-10
        b = self.hyper_b(state)
        q_t = w * qs + b
        v_t = w * max_qs + b
        adv = q_t - v_t
        key = self.key(state).abs().view(B, self.H, 1) + 1e-10
        ag = torch.sigmoid(self.agents(state)).view(B, self.H, self.n)
        ac = torch.sigmoid(self.action(torch.cat([state, actions_onehot.reshape(B, -1)], -1))).view(B, self.H, self.n)
        lam = (key * ag * ac).sum(1)
        return q_t.sum(-1) + ((lam - 1) * adv.detach()).sum(-1)
