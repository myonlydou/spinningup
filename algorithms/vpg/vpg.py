import csv
from pathlib import Path

import torch
import numpy as np
import gymnasium as gym


from gymnasium.spaces import Discrete, Box

from core import MLPActorCritic


class VanillaPolicyGradiengtBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma, lam) -> None:
        self.obs_buf = np.zeros(self._combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(self._combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)

        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.gamma, self.lam = gamma, lam

    def store(self, obs, act, rew, val, logp):
        ptr = self.ptr
        assert ptr < self.max_size
        self.ptr += 1

        self.obs_buf[ptr] = obs
        self.act_buf[ptr] = act
        self.rew_buf[ptr] = rew
        self.val_buf[ptr] = val
        self.logp_buf[ptr] = logp

    def finish_episode(self, last_val):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # δ_t = r_t + γV(s_{t+1}) − V(s_t)；优势 = discount_cumsum(δ, γλ)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        # 计算 GAE Advanture
        self.adv_buf[path_slice] = self.discount_cumsum(deltas, self.gamma * self.lam)

        # 价值网络的回归目标：reward-to-go，末尾多算的一项丢掉
        self.ret_buf[path_slice] = self.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self, device):
        assert self.ptr == self.max_size, "buffer 还没存满"
        self.ptr, self.path_start_idx = 0, 0

        adv = self.adv_buf
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        data = dict(
            obs=self.obs_buf,
            act=self.act_buf,
            ret=self.ret_buf,
            adv=adv.astype(np.float32),
            logp=self.logp_buf,
        )

        return {
            k: torch.as_tensor(v, dtype=torch.float32, device=device)
            for k, v in data.items()
        }

    def discount_cumsum(self, x, discount):
        """反向累计折扣和：[x0 + γx1 + γ²x2, x1 + γx2, x2]。"""
        x = np.asarray(x, dtype=np.float64)
        out = np.zeros_like(x)
        acc = 0.0
        for t in reversed(range(len(x))):
            acc = x[t] + discount * acc
            out[t] = acc
        return out

    def _combined_shape(self, length, shape=None):
        """把 (length, shape) 拼成缓冲区数组的维度，shape 为 () 时退化成 (length,)。"""
        if shape is None:
            return (length,)
        if isinstance(shape, (int, np.integer)):
            return (length, int(shape))
        return (length, *shape)


class VanillaPolicyGradient:
    def __init__(
        self,
        env_id,
        hidden_sizes,
        pi_lr,
        vf_lr,
        gamma,
        lam,
        train_v_iters,
        steps_per_epoch,
        device,
        seed,
    ):
        env = gym.make(env_id, render_mode=None)
        assert isinstance(
            env.observation_space, Box
        ), "This example only works for envs with continuous state spaces."
        assert isinstance(
            env.action_space, (Box, Discrete)
        ), "This example only works for envs with continuous action spaces."

        obs_space = env.observation_space
        act_space = env.action_space

        self.env = env
        self.env_id = env_id

        self.pi_lr = pi_lr
        self.vf_lr = vf_lr
        self.train_v_iters = train_v_iters
        self.steps_per_epoch = steps_per_epoch
        self.device = device

        self.ac = MLPActorCritic(
            obs_space,
            act_space,
            hidden_sizes,
        ).to(device)

        self.pi_optimizer = torch.optim.Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.v_optimizer = torch.optim.Adam(self.ac.v.parameters(), vf_lr)

        self.buffer = VanillaPolicyGradiengtBuffer(
            obs_space.shape, act_space.shape, steps_per_epoch, gamma, lam
        )

        # Random seed
        torch.manual_seed(seed)
        np.random.seed(seed)

    def _as_tensor(self, obs):
        return torch.as_tensor(obs, dtype=torch.float32, device=self.device)

    def _loss_pi(self, data):
        obs, act, adv = data["obs"], data["act"], data["adv"]
        _, logp = self.ac.pi(obs, act)
        return -(logp * adv).mean()

    def _loss_v(self, data):
        obs, ret = data["obs"], data["ret"]
        return ((self.ac.v(obs) - ret) ** 2).mean()

    @torch.no_grad()
    def _diagnostics(self, data):
        """更新之后调用，看这次更新把策略推了多远、熵还剩多少。"""
        pi, logp = self.ac.pi(data["obs"], data["act"])
        kl = (data["logp"] - logp).mean().item()  # 近似 KL(旧‖新)，仅监控
        ent = pi.entropy().mean().item()
        return kl, ent

    def _update_net(self):
        data = self.buffer.get(self.device)

        with torch.no_grad():
            loss_pi_old = self._loss_pi(data)
            loss_v_old = self._loss_v(data)

        self.pi_optimizer.zero_grad()
        loss_pi = self._loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        kl, ent = self._diagnostics(data)

        loss_v = loss_v_old.detach().clone()
        for _ in range(self.train_v_iters):
            self.v_optimizer.zero_grad()
            loss_v = self._loss_v(data)
            loss_v.backward()
            self.v_optimizer.step()

        return {
            "loss_pi": loss_pi_old,
            "loss_v": loss_v_old,
            "delta_loss_v": loss_v.item() - loss_v_old,
            "kl": kl,
            "ent": ent,
        }

    def train_one_epoch(self):
        obs, _ = self.env.reset()

        ep_ret, ep_len = 0.0, 0
        ep_rets, ep_lens = [], []
        for t in range(self.steps_per_epoch):
            act, val, logp = self.ac.step(self._as_tensor(obs))

            next_obs, rew, terminated, truncated, _ = self.env.step(act)

            self.buffer.store(obs, act, rew, val, logp)

            ep_ret += float(rew)
            ep_len += 1

            obs = next_obs

            done = terminated or truncated
            epoch_ended = t == self.steps_per_epoch - 1
            if done or epoch_ended:
                last_val = 0.0

                if not terminated:
                    last_val = self.ac.step(self._as_tensor(obs))[1]

                self.buffer.finish_episode(last_val)

                if done:
                    ep_rets.append(ep_ret)
                    ep_lens.append(ep_len)

                obs, _ = self.env.reset()
                ep_ret, ep_len = 0.0, 0

        stats = self._update_net()
        stats["ep_ret"] = float(np.mean(ep_rets)) if ep_rets else float("nan")
        stats["ep_len"] = float(np.mean(ep_lens)) if ep_lens else float("nan")

        return stats

    def play(self):
        env = gym.make(self.env_id, render_mode="human")
        obs, _ = env.reset()
        done = False

        ep_ret = 0.0
        ep_len = 0
        while not done:
            act = self.ac.act(self._as_tensor(obs))
            obs, rew, terminated, truncated, _ = env.step(act)

            done = terminated or truncated
            ep_ret += float(rew)
            ep_len += 1

        env.close()
        return ep_ret, ep_len


def run_vpg(epochs=50, save=None):
    """命令行入口：训练若干个 epoch，可选把每轮统计量存成 CSV，最后渲染一局。"""
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", "--env", type=str, default="HalfCheetah-v5")
    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--steps_per_epoch", type=int, default=4000)
    parser.add_argument("--pi_lr", type=float, default=1e-2)
    parser.add_argument("--vf_lr", type=float, default=1e-2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.97)
    parser.add_argument("--train_v_iters", type=int, default=80)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save",
        "-s",
        nargs="?",
        const="auto",
        default=save,
        help="把每轮统计量存成 CSV；只写 --save 不给路径时存到 "
        "data/vpg_<env>_seed<seed>.csv",
    )
    args = parser.parse_args()

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 小型的矩阵在cpu上更快
    device = torch.device("cpu")

    agent = VanillaPolicyGradient(
        env_id=args.env_name,
        hidden_sizes=[args.hidden] * args.layers,
        pi_lr=args.pi_lr,
        vf_lr=args.vf_lr,
        steps_per_epoch=args.steps_per_epoch,
        gamma=args.gamma,
        lam=args.lam,
        train_v_iters=args.train_v_iters,
        device=device,
        seed=args.seed,
    )

    # 统计量要落盘时，先确定路径（auto 表示按环境和种子自动命名）
    csv_path = None
    if args.save is not None:
        if args.save == "auto":
            csv_path = Path("data") / f"vpg_{args.env_name}_seed{args.seed}.csv"
        else:
            csv_path = Path(args.save)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    csv_file, writer = None, None
    try:
        for epoch in range(1, args.epochs + 1):
            stats = agent.train_one_epoch()
            print(
                "epoch: %3d'th \t return: %6.1f \t ep_len: %5.1f \t "
                "loss_pi: %+8.3f \t loss_v: %8.3f \t kl: %+7.4f \t ent: %.3f \t %.1fs"
                % (
                    epoch,
                    stats["ep_ret"],
                    stats["ep_len"],
                    stats["loss_pi"],
                    stats["loss_v"],
                    stats["kl"],
                    stats["ent"],
                    time.time() - start,
                )
            )

            if csv_path is not None:
                # 列名取自第一轮的 stats，避免以后加指标时忘了同步表头
                row = {"epoch": epoch, **{k: float(v) for k, v in stats.items()}}
                if writer is None:
                    preferred = [
                        "ep_ret",
                        "ep_len",
                        "loss_pi",
                        "loss_v",
                        "delta_loss_v",
                        "kl",
                        "ent",
                    ]
                    fields = ["epoch"]
                    fields += [k for k in preferred if k in row]
                    fields += [k for k in row if k not in fields]
                    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
                    writer = csv.DictWriter(csv_file, fieldnames=fields)
                    writer.writeheader()
                writer.writerow(row)
                if csv_file is not None:
                    csv_file.flush()  # 中途 Ctrl-C 也能留下已经跑完的轮次
    finally:
        if csv_file is not None:
            csv_file.close()

    if csv_path is not None:
        print(f"统计量已保存: {csv_path}")

    ret, ep_len = agent.play()

    print("return: %3f, len:%3d" % (ret, ep_len))

    agent.env.close()


if __name__ == "__main__":
    run_vpg()
