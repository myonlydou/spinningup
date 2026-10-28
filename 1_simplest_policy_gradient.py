"""最简单的策略梯度：整条轨迹的总回报作为每个 step 的权重。

权重只和"这一局总共拿多少分"有关，和该 step 在轨迹里的位置无关。
"""

from policy_gradient_core import PolicyGradient, run


class SimplestPolicyGradient(PolicyGradient):
    def _weights(self, ep_rews):
        return [sum(ep_rews)] * len(ep_rews)


if __name__ == "__main__":
    run(SimplestPolicyGradient)
