# CampaignPulse 分渠道精确推断（纯 Python 标准库）

对营销 A/B 报告做**分渠道精确条件推断**，避免把渠道构成变化误判为实验效果。
不合并渠道、不平均各渠道检验，也不使用正态/卡方/模拟近似。

## 输入

每个渠道一个四元组 `(nA, nB, xA, xB)`：

- `nA, nB`：A、B 两臂访客数（去重后，每人只属一个渠道、一个臂）
- `xA, xB`：A、B 两臂转化人数

最多 5 个渠道，每臂至少 1 名访客，输入保证合法。

## 统计定义

固定每个渠道两臂访客数与总转化人数 `m = xA + xB`。共同优势比为
`theta` 时，B 臂转化人数 `K` 的条件权重为

```
C(nB, k) * C(nA, m-k) * theta**k
```

在使两个组合数均非零的整数 `k` 上归一化得到该渠道分布；各渠道独立，
`T = ΣK`，观测 `t = ΣxB`。`theta > 1` 表示 B 优于 A。

- `evaluate(strata, theta)`：返回 `T` 的期望，以及观测值两侧含等号尾概率
  `2*min(P(T≤t), P(T≥t))`（截到 1）的自然对数。
- `infer(strata)`：返回共同优势比条件 MLE、95% 置信区间、`theta=1` 时的
  双侧 p 值及其自然对数。
  - MLE：解 `E[T]=t`；
  - 置信区间下端点解 `P(T≥t)=0.025`，上端点解 `P(T≤t)=0.025`；
  - 观测在支持集端点时相应估计/端点为 `0.0` 或 `math.inf`；
  - 支持集只有一个值（全无信息）时估计为 `None`（不可识别），
    区间 `[0, +∞]`，p 值为 1。

## 用法

```python
import math
from campaign_pulse import evaluate, infer

strata = [
    (1000, 100, 900, 96),   # nA, nB, xA, xB
    (100, 1000, 30, 340),
]

ev = evaluate(strata, theta=1.0)
print(ev.expected, ev.log_two_sided)

r = infer(strata)
print(r.odds_ratio, r.ci_low, r.ci_high, r.p_value, r.log_p_value)
# odds_ratio 为 None 表示不可识别；端点可能是 0.0 或 math.inf。
# p 值极小时可下溢为 0.0，但 r.log_p_value 保持有限且正确。
```

## 实现要点

- 每渠道只枚举合法 `k`（宽度 ≤ 2000 总量），用**对数空间卷积**合并，
  生产路径不枚举渠道笛卡尔积、不展开逐访客数据。
- 卷积只做一次；之后用**指数倾斜**以 O(宽度) 计算任意 `theta` 下的
  归一化常数、期望与双侧尾概率。
- 每个对数和都以**自身局部峰值**累加，远离全局峰的尾概率也不会下溢，
  因此 `log_p_value` 在 p≈e^-1000 量级仍精确。
- 根求解用括号倍增 + 二分（固定 80 次），全部仅用 `math` 标准库。

## 测试

```
python3 -m unittest test_campaign_pulse -v
```

覆盖：小样本笛卡尔积穷举独立参照（尾概率 / MLE / 区间反演）、零转化渠道、
全无信息数据、支持集端点观测、无信息渠道不改变推断、Simpson 悖论
（逐渠道 B 更优但合并后反转，共同 OR 仍 >1）、以及 5 渠道 / 50 万访客 /
宽度和 2000 / 近端点（`log p < -1000`、端点有限）的 10 秒内性能测试。
