"""策略梯度算法的公共部分：MLP、训练循环、更新逻辑、渲染测试。

各个编号脚本只需要继承 PolicyGradient 并实现 _weights()，
用它说明"每个 step 的权重怎么算"这一处差异：

- 1_simplest_policy_gradient.py   整条轨迹的回报
- 2_reward_to_go_policy_gradient.py  reward-to-go
"""

import argparse

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box, Discrete
from torch.distributions.categorical import Categorical


class MLP(nn.Module):
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


class PolicyGradient:
    """策略梯度基类，采样、更新、训练循环、渲染测试都在这里。"""

    def __init__(
        self,
        env_id,
        hidden_sizes,
        lr,
        batch_size,
        device,
    ):

        env = gym.make(env_id, render_mode=None)

        assert isinstance(
            env.observation_space, Box
        ), "This example only works for envs with continuous state spaces."
        assert isinstance(
            env.action_space, Discrete
        ), "This example only works for envs with discrete action spaces."

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n

        self.policy_net = MLP([obs_dim] + hidden_sizes + [act_dim]).to(device)

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

        self.env = env
        self.env_id = env_id
        self.batch_size = batch_size
        self.device = device

    # ---------- 子类唯一需要改写的地方 ----------
    def _weights(self, ep_rews):
        """返回一条完整轨迹里每个 step 的权重，长度与 ep_rews 相同。"""
        raise NotImplementedError

    # ---------- 以下逻辑各版本共用 ----------
    def _policy(self, obs):
        logits = self.policy_net(obs)
        return Categorical(logits=logits)

    def _sample_action(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        return self._policy(obs).sample().item()

    def _update_policy(self, obs_batch, act_batch, weights_batch):
        obs = torch.as_tensor(
            np.asarray(obs_batch), dtype=torch.float32, device=self.device
        )
        acts = torch.as_tensor(act_batch, dtype=torch.int32, device=self.device)
        weights = torch.as_tensor(
            np.asarray(weights_batch), dtype=torch.float32, device=self.device
        )

        self.optimizer.zero_grad()
        logp = self._policy(obs).log_prob(acts)
        loss = -(logp * weights).mean()
        loss.backward()
        self.optimizer.step()

        return loss

    def train_one_epoch(self):
        obs_batch = []
        act_batch = []
        weight_batch = []
        ret_batch = []
        len_batch = []

        obs, _ = self.env.reset()
        done = False

        ep_rews = []

        while not done:

            obs_batch.append(obs.copy())

            act = self._sample_action(obs)
            obs, rew, terminated, truncated, _ = self.env.step(act)

            act_batch.append(act)
            ep_rews.append(rew)

            done = terminated or truncated

            if done:
                ep_ret = sum(ep_rews)
                ep_len = len(ep_rews)

                ret_batch.append(ep_ret)
                len_batch.append(ep_len)

                weight_batch += list(self._weights(ep_rews))

                obs, _ = self.env.reset()
                ep_rews = []

                if len(obs_batch) < self.batch_size:
                    done = False

        loss = self._update_policy(obs_batch, act_batch, weight_batch)

        return loss.item(), ret_batch, len_batch

    def play(self):
        env = gym.make(self.env_id, render_mode="human")
        obs, _ = env.reset()
        done = False

        ep_ret = 0
        ep_len = 0
        while not done:
            act = self._sample_action(obs)
            obs, rew, terminated, truncated, _ = env.step(act)

            done = terminated or truncated
            ep_ret += float(rew)
            ep_len += 1

        env.close()
        return ep_ret, ep_len


def run(policy_cls, epochs=50):
    """命令行入口：训练若干个 epoch，再渲染一局看看效果。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", "--env", type=str, default="CartPole-v1")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=epochs)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = policy_cls(
        env_id=args.env_name,
        hidden_sizes=[32],
        lr=args.lr,
        batch_size=5000,
        device=device,
    )

    for i in range(args.epochs):
        batch_loss, batch_rets, batch_lens = agent.train_one_epoch()
        print(
            "epoch: %3d'th \t loss: %.3f \t return: %.3f \t ep_len: %.3f"
            % (i + 1, batch_loss, np.mean(batch_rets), np.mean(batch_lens))
        )

    ret, ep_len = agent.play()

    print("return: %3f, len:%3d" % (ret, ep_len))

    agent.env.close()
