import torch.nn as nn
import gymnasium as gym
import numpy as np
import torch

from gymnasium.spaces import Discrete, Box
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


class Reward2GoPolicyGradient:
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

    def _reward_to_go(self, rews):
        n = len(rews)
        rtgs = np.zeros_like(rews)
        for i in reversed(range(n)):
            rtgs[i] = rews[i] + (rtgs[i + 1] if i + 1 < n else 0)
        return rtgs

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

                weight_batch += list(self._reward_to_go(ep_rews))

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", "--env", type=str, default="CartPole-v1")
    parser.add_argument("--lr", type=float, default=1e-2)
    args = parser.parse_args()

    epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    r2gpg = Reward2GoPolicyGradient(
        env_id=args.env_name,
        hidden_sizes=[32],
        lr=args.lr,
        batch_size=5000,
        device=device,
    )

    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = r2gpg.train_one_epoch()
        print(
            "epoch: %3d'th \t loss: %.3f \t return: %.3f \t ep_len: %.3f"
            % (i + 1, batch_loss, np.mean(batch_rets), np.mean(batch_lens))
        )

    ret, ep_len = r2gpg.play()

    print("return: %3f, len:%3d" % (ret, ep_len))

    r2gpg.env.close()
