import torch

import numpy as np
import torch.nn as nn

from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from gymnasium.spaces import Discrete, Box


class MLP(nn.Module):
    """
    创建一个 MLP
    激活函数默认为 nn.Tanh
    """

    def __init__(
        self,
        sizes,
        activation=nn.Tanh,
        output_activation=nn.Identity,
    ):
        super().__init__()

        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))

            if i < len(sizes) - 2:
                layers.append(activation())
            else:
                layers.append(output_activation())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        """不给 act 时只返回分布；给了 act 就顺带算它的 log 概率。"""
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.Tanh):
        super().__init__()
        self.logits_net = MLP([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        return Categorical(logits=self.logits_net(obs))

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.Tanh):
        super().__init__()
        # 与状态无关的可学习 log 标准差，真正的策略网络只输出均值
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = MLP([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        # Normal 各维独立，要对动作维求和才得到每个样本一个标量
        return pi.log_prob(act).sum(axis=-1)


class MLPCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation=nn.Tanh):
        super().__init__()
        self.v_net = MLP([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        # (batch, 1) -> (batch,)
        return torch.squeeze(self.v_net(obs), -1)


class MLPActorCritic(nn.Module):

    def __init__(
        self,
        obs_space,
        act_space,
        hidden_sizes,
        activation=nn.Tanh,
    ) -> None:
        super().__init__()

        obs_dim = obs_space.shape[0]

        if isinstance(act_space, Box):
            self.pi = MLPGaussianActor(
                obs_dim, act_space.shape[0], hidden_sizes, activation
            )
        elif isinstance(act_space, Discrete):
            self.pi = MLPCategoricalActor(
                obs_dim, act_space.n, hidden_sizes, activation
            )
        else:
            raise NotImplementedError(f"不支持的动作空间类型: {type(act_space)}")

        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

    def act(self, obs):
        return self.step(obs)[0]
