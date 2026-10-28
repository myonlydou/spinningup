"""reward-to-go 版策略梯度：每个 step 只用它之后拿到的那部分回报。

相比 1_simplest_policy_gradient.py，奖励里少了"step 之前已经发生"的部分，
梯度估计的方差更小。两个版本唯一的区别就是这里的 _weights()。
"""

import numpy as np

from policy_gradient_core import PolicyGradient, run


class RewardToGoPolicyGradient(PolicyGradient):
    def _reward_to_go(self, rews):
        n = len(rews)
        r2gs = np.zeros_like(rews)
        for i in reversed(range(n)):
            r2gs[i] = rews[i] + (r2gs[i + 1] if i + 1 < n else 0)
        return r2gs

    def _weights(self, ep_rews):
        return self._reward_to_go(ep_rews)


if __name__ == "__main__":
    run(RewardToGoPolicyGradient)
