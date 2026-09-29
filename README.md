# CampaignPulse 分渠道精确推断

纯 Python 标准库实现的营销 A/B 实验分渠道（分层）精确推断组件。
在各渠道固定两臂访客数与总转化数的条件下，使用非中心超几何分布，
并对各渠道的离散分布在**对数空间**做精确卷积，得到总 B 臂转化人数
`T` 的精确分布，从而对共同优势比（common odds ratio）做推断。
不合并渠道、不做正态/卡方/模拟近似，渠道构成变化不会被误判为实验效果。

## 输入

每个渠道一个 `(nA, nB, xA, xB)` 四元组（或含同名键的 dict）：
A/B 两臂访客数与转化人数。访客已去重且唯一归属一个渠道与一个实验臂。

## API

```python
from campaignpulse import evaluate, infer

strata = [(100, 10, 20, 3), (10, 100, 1, 15)]

# 任意 theta（有限正数）下 T 的期望与观测值两侧含等号尾概率的自然对数
r = evaluate(strata, theta=1.0)
r.expectation   # E_theta[T]
r.log_tail_ge   # log P_theta(T >= t_obs)
r.log_tail_le   # log P_theta(T <= t_obs)

# 共同优势比推断（> 1 表示 B 优于 A）
r = infer(strata)
r.estimate      # 使 E_theta[T] = t_obs 的 theta；支持集下端为 0、上端为 inf
r.ci_lower      # 95% CI 下端：P(T >= t_obs) = 0.025
r.ci_upper      # 95% CI 上端：P(T <= t_obs) = 0.025
r.p_value       # 双侧 p 值：2 * min(两侧尾概率)，截到 1（可因下溢为 0）
r.log_p         # p 值的自然对数，始终保持有限且精确
```

特殊情况：

- 整个支持集只有一个值（所有渠道都无信息）时，`estimate` 为 `None`
  （不可识别），区间为 `(0.0, inf)`，`p_value` 为 `1.0`。
- 观测落在支持集端点时，对应侧的估计/区间端点为 `0.0` 或 `inf`，
  另一侧仍按定义求解。
- 无信息渠道（合法 k 唯一）自动不影响其他渠道的推断。

## 测试

```bash
python3 -m unittest -v
```

测试含小样本笛卡尔积穷举参照（尾概率、估计、区间反演）、零转化与
无信息渠道、支持集端点观测、Simpson 悖论（各渠道 B 优、合并 B 劣时
共同优势比仍 > 1），以及 5 渠道 / 50 万访客 / 宽度之和 2000、
`log p < -1000` 的最大规模性能用例（单次 `infer` 远低于 10 秒）。
