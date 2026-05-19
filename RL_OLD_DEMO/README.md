# LunarLander 强化学习入门指南

## 目录
1. [什么是 LunarLander？](#1-什么是-lunarlander)
2. [强化学习基础概念](#2-强化学习基础概念)
3. [环境详解](#3-环境详解)
4. [安装与环境配置](#4-安装与环境配置)
5. [Quick Start：让飞船随机飞](#5-quick-start让飞船随机飞)
6. [训练一个 DQN 智能体](#6-训练一个-dqn-智能体)
7. [训练一个 PPO 智能体](#7-训练一个-ppo-智能体)
8. [训练一个 SAC 智能体（连续动作版）](#8-训练一个-sac-智能体连续动作版)
9. [Safe PPO：追求零坠毁](#9-safe-ppo追求零坠毁)
10. [四种算法对比](#10-四种算法对比)
11. [理解训练输出](#11-理解训练输出)
12. [理解训练过程](#12-理解训练过程)
13. [常见问题](#13-常见问题)

---

## 1. 什么是 LunarLander？

LunarLander 是 **Gymnasium**（原 OpenAI Gym）提供的一个经典强化学习环境。

你需要训练一个 AI 智能体，控制一艘月球飞船，让它**安全降落**在地面的两个旗子之间。

```
        ___
       /   \       ← 飞船
      | o o |
       \___/
      /  |  \
     /   |   \
    /    ↓    \
   /  发动机   \
  _______________
 /               \
/  着陆垫 🚩🚩  \   ← 目标区域
```

### 两种版本

| 版本 | 动作类型 | 说明 |
|------|----------|------|
| `LunarLander-v3` | 离散（4个动作） | 初学者推荐 |
| `LunarLanderContinuous-v3` | 连续（2个浮点数） | 进阶使用 |

---

## 2. 强化学习基础概念

在正式使用之前，先理解几个核心概念：

```
环境 (Environment)
    ↓ 观测状态 (Observation)
智能体 (Agent)
    ↓ 动作 (Action)
环境 (Environment)
    ↓ 奖励 (Reward) + 新状态
智能体 (Agent)
    ↓ 学习更新 ...
```

| 概念 | 在 LunarLander 中的含义 |
|------|------------------------|
| **环境 (Env)** | 月球、重力、飞船物理模拟 |
| **智能体 (Agent)** | 你训练的 AI 模型 |
| **状态 (State/Obs)** | 飞船当前位置、速度、角度等 8 个数值 |
| **动作 (Action)** | 开哪个发动机（或不开） |
| **奖励 (Reward)** | 成功降落 +200，坠毁 -100，等 |
| **回合 (Episode)** | 从起飞到落地（或坠毁）的一次完整过程 |

---

## 3. 环境详解

### 3.1 状态空间（Observation Space）

每一步，环境返回一个 **8维向量**：

```
obs = [x, y, vx, vy, angle, angular_velocity, left_leg, right_leg]
```

| 索引 | 变量名 | 含义 | 范围 |
|------|--------|------|------|
| 0 | `x` | 飞船水平位置，0 为着陆垫中心，负数偏左，正数偏右 | -1.5 ~ 1.5 |
| 1 | `y` | 飞船垂直高度，0 为地面，越大越高 | -1.5 ~ 1.5 |
| 2 | `vx` | 水平速度，负数向左飞，正数向右飞 | -5 ~ 5 |
| 3 | `vy` | 垂直速度，负数下降，正数上升 | -5 ~ 5 |
| 4 | `angle` | 飞船倾斜角度，0 为竖直，正数向右倾，负数向左倾 | -π ~ π |
| 5 | `angular_velocity` | 旋转速度，正数顺时针，负数逆时针 | -5 ~ 5 |
| 6 | `left_leg` | 左支脚是否触地，1 = 已接触地面，0 = 未接触 | 0 或 1 |
| 7 | `right_leg` | 右支脚是否触地，1 = 已接触地面，0 = 未接触 | 0 或 1 |

理想的着陆状态：`x ≈ 0`（对准中心）、`vy ≈ 0`（缓慢下降）、`angle ≈ 0`（保持竖直）、`left_leg = right_leg = 1`（双脚触地）。

### 3.2 动作空间（Action Space）

离散版共 **4 个动作**：

| 动作编号 | 含义 |
|----------|------|
| 0 | 什么都不做（关闭所有发动机） |
| 1 | 启动左侧发动机 |
| 2 | 启动主发动机（向下推力） |
| 3 | 启动右侧发动机 |

### 3.3 奖励机制（Reward）

| 情况 | 奖励 |
|------|------|
| 飞近着陆垫 | 正奖励 |
| 飞远着陆垫 | 负奖励 |
| 支脚接触地面 | +10 每条腿 |
| 成功着陆 | +100 ~ +140 |
| 坠毁 | -100 |
| 每次启动主发动机 | -0.3（鼓励省油） |
| 每次启动侧发动机 | -0.03 |

> **目标**：单回合累计奖励达到 **200 分以上**视为成功。

---

## 4. 安装与环境配置

### 4.1 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv rl_env
source rl_env/bin/activate      # macOS/Linux
# rl_env\Scripts\activate       # Windows

# 安装核心库
pip install gymnasium[box2d]    # 环境本体
pip install stable-baselines3   # 封装好的 RL 算法库
pip install torch               # PyTorch（SB3 后端）
pip install matplotlib          # 绘图
```

> **注意**：`box2d` 是 LunarLander 的物理引擎依赖，必须安装。

### 4.2 验证安装

```python
import gymnasium as gym

env = gym.make("LunarLander-v3")
obs, info = env.reset()
print("状态空间维度:", env.observation_space.shape)  # (8,)
print("动作空间大小:", env.action_space.n)            # 4
print("初始状态:", obs)
env.close()
print("安装成功！")
```

---

## 5. Quick Start：让飞船随机飞

在训练任何模型之前，先用**随机策略**跑一下，熟悉 API：

```python
import gymnasium as gym

# 创建环境，render_mode="human" 会弹出可视化窗口
env = gym.make("LunarLander-v3", render_mode="human")

obs, info = env.reset(seed=42)
total_reward = 0

for step in range(500):
    # 随机选一个动作（0~3）
    action = env.action_space.sample()
    
    # 执行动作，获取新状态
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    # 回合结束（降落或坠毁）
    if terminated or truncated:
        print(f"回合结束，总奖励: {total_reward:.1f}")
        break

env.close()
```

随机策略的得分通常在 **-200 ~ 0** 之间，飞船会乱飞然后坠毁，这是正常的。

---

## 6. 训练一个 DQN 智能体

### 6.1 DQN 是什么？

**DQN（Deep Q-Network）** 是最经典的深度强化学习算法之一：

```
状态 (8维) → 神经网络 → 每个动作的 Q 值 → 选最大值的动作
```

Q 值可以理解为：**"在当前状态下执行这个动作，未来能获得多少总奖励"**

### 6.2 使用 Stable-Baselines3 训练

Stable-Baselines3（SB3）封装了常用的 RL 算法，几行代码就能训练：

```python
import gymnasium as gym
from stable_baselines3 import DQN

# 1. 创建环境
env = gym.make("LunarLander-v3")

# 2. 创建 DQN 模型
model = DQN(
    policy="MlpPolicy",   # 多层感知机（全连接神经网络）
    env=env,
    learning_rate=1e-3,   # 学习率
    buffer_size=50000,    # 经验回放池大小
    verbose=1             # 打印训练信息
)

# 3. 训练（约需 5~10 分钟）
model.learn(total_timesteps=1_000_000)

# 4. 保存模型
model.save("lunarlander_dqn")
print("训练完成，模型已保存！")
env.close()
```

### 6.3 加载并测试训练好的模型

```python
import gymnasium as gym
from stable_baselines3 import DQN

# 加载模型
model = DQN.load("lunarlander_dqn")

# 可视化测试
env = gym.make("LunarLander-v3", render_mode="human")
obs, info = env.reset()

total_reward = 0
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    if terminated or truncated:
        print(f"最终得分: {total_reward:.1f}")
        break

env.close()
```

---

## 7. 训练一个 PPO 智能体

### 7.1 PPO 是什么？

**PPO（Proximal Policy Optimization）** 是目前最流行的深度强化学习算法之一，由 OpenAI 于 2017 年提出。

与 DQN 不同，PPO 直接学习**策略函数**（给定状态输出动作概率），而不是 Q 值：

```
状态 (8维) → 神经网络 → 每个动作的概率分布 → 采样动作
```

PPO 的核心思想是**限制每次更新的幅度**，防止策略突变导致训练崩溃，所以比 DQN 稳定得多。

### 7.2 PPO 的关键优势

- **支持多进程并行采样**：16 个环境同时跑，数据收集速度快 16 倍
- **on-policy 算法**：每批数据来自当前策略，更新更稳定
- **收敛更快**：同等步数下通常比 DQN 得分更高

### 7.3 使用 Stable-Baselines3 训练（16 进程）

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

# 16 个并行环境
env = make_vec_env("LunarLander-v3", n_envs=16, vec_env_cls=SubprocVecEnv)

model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=1024,      # 每个进程每轮采集步数
    batch_size=256,
    n_epochs=10,
    gamma=0.999,
    gae_lambda=0.98,
    ent_coef=0.01,     # 熵正则，鼓励探索
    verbose=1,
)

model.learn(total_timesteps=4_000_000)
model.save("final_ppo")
env.close()
```

> 4M 步 × 16 进程 = 实际采样 **6400 万**次环境交互，通常能稳定收敛到 200 分以上。

### 7.4 运行 Demo

```bash
python demo_ppo.py --mode train     # 训练
python demo_ppo.py --mode test      # 测试最佳模型
```

训练结束后会生成：
- `best_ppo.zip` — 训练过程中得分最高的模型
- `final_ppo.zip` — 最终模型
- `training_curve_ppo.png` — 训练曲线

---

## 8. 训练一个 SAC 智能体（连续动作版）

### 8.1 连续动作版的区别

离散版只能选 4 个动作之一，连续版输出 **2 个浮点数**，可以精细控制油门：

```
obs = [x, y, vx, vy, angle, angular_velocity, left_leg, right_leg]  ← 状态（相同）

action = [main_engine, lateral_engine]
           ↑                ↑
        主引擎油门        侧引擎油门
        -1~0: 关闭       -1~0: 左引擎
        0~1: 推力        0~1:  右引擎
```

连续版更难控制，但飞行更流畅，更接近真实控制问题。

### 8.2 SAC 是什么？

**SAC（Soft Actor-Critic）** 是连续控制领域最主流的算法：

- **最大化熵**：在获得高奖励的同时，尽量保持动作的多样性，避免陷入局部最优
- **off-policy**：可以反复利用历史数据，采样效率高
- **自动调整探索**：`ent_coef="auto"` 会自动调整探索强度，不需要手动调参

### 8.3 使用 Stable-Baselines3 训练

```python
from stable_baselines3 import SAC
import gymnasium as gym

env = gym.make("LunarLanderContinuous-v3")

model = SAC(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    buffer_size=200_000,
    batch_size=256,
    ent_coef="auto",     # 自动调整熵系数
    verbose=1,
)

model.learn(total_timesteps=500_000)
model.save("final_sac")
env.close()
```

### 8.4 运行 Demo

```bash
python demo_sac.py --mode train     # 训练
python demo_sac.py --mode test      # 测试最佳模型
```

训练结束后生成：
- `best_sac.zip` — 训练过程中得分最高的模型
- `final_sac.zip` — 最终模型
- `training_curve_sac.png` — 训练曲线

---

## 9. Safe PPO：追求零坠毁

### 9.1 为什么普通 PPO 还会坠毁？

普通 PPO 的目标是**最大化平均奖励**，偶尔坠毁换来的 -100 在统计上是可以接受的损失。
它不会主动规避坠毁，只是碰巧大多数时候能降落。

### 9.2 Safe PPO 的核心思路：奖励塑形

通过修改奖励函数，让坠毁的代价大到模型**主动不愿意冒险**：

```
原始奖励
  + 坠毁额外惩罚       -500   (原本只有 -100)
  + 低空超速惩罚       按超速量持续扣分
  + 大角度惩罚         倾斜超过 17° 持续扣分
  + 平稳着陆奖励       双脚稳落 +50
= 安全奖励
```

用 `gymnasium.Wrapper` 包装环境，不需要改算法本身：

```python
class SafetyRewardWrapper(Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        x, y, vx, vy, angle, ang_vel, left_leg, right_leg = obs

        # 坠毁额外惩罚
        if terminated and not (left_leg or right_leg):
            reward += -500.0

        # 低空高速惩罚
        if y < 0.5 and vy < -1.2:
            reward += -2.0 * (abs(vy) - 1.2)

        # 大角度惩罚
        if abs(angle) > 0.3:
            reward += -1.5 * (abs(angle) - 0.3)

        # 平稳着陆奖励
        if left_leg and right_leg and abs(vy) < 0.5 and abs(angle) < 0.15:
            reward += 50.0

        return obs, reward, terminated, truncated, info
```

### 9.3 PPO 超参数也更保守

| 参数 | 普通 PPO | Safe PPO | 原因 |
|------|----------|----------|------|
| `learning_rate` | 3e-4 | 2e-4 | 更小步长，更稳 |
| `ent_coef` | 0.01 | 0.005 | 减少随机探索 |
| `clip_range` | 0.2 | 0.15 | 每步更新更保守 |
| `n_steps` | 1024 | 2048 | 更长视野 |

### 9.4 运行 Demo

```bash
python demo_safe.py --mode train             # 训练（默认 10M 步，全部 CPU 核心）
python demo_safe.py --mode test              # 测试 50 回合，统计坠毁率
python demo_safe.py --mode test --n-episodes 100   # 测试 100 回合
```

训练结束后生成：
- `best_safe.zip` — 训练过程中得分最高的模型
- `final_safe.zip` — 最终模型
- `training_curve_safe.png` — 训练曲线

> 注意：训练曲线的奖励包含安全惩罚，会比普通 PPO 的数值低，这是正常的。
> 测试时用的是**原始环境**，展示真实得分和坠毁率。

---

## 10. 四种算法对比

| 对比项 | DQN | PPO | SAC | Safe PPO |
|--------|-----|-----|-----|----------|
| 动作类型 | 离散 | 离散/连续 | 连续 | 离散 |
| 算法类型 | Off-policy | On-policy | Off-policy | On-policy |
| 多进程支持 | 不适合 | 显著加速 | 不适合 | 显著加速 |
| 收敛速度 | 慢 | 中 | 快 | 慢（步数多） |
| 稳定性 | 一般 | 好 | 很好 | 最好 |
| 安全性 | 低 | 中 | 中 | 高 |
| 核心机制 | Q 值学习 | 裁剪策略梯度 | 最大熵策略 | 奖励塑形 + PPO |
| 推荐训练步数 | 1,000,000 | 4,000,000 | 500,000 | 10,000,000 |
| 适用场景 | 入门学习 | 离散/需要快速训练 | 连续控制 | 追求零坠毁 |

**选哪个？**
- 刚入门 → DQN（概念最清晰）
- 离散动作、追求速度 → PPO + 多进程
- 连续动作 → SAC

---

| 对比项 | DQN | PPO |
|--------|-----|-----|
| 算法类型 | Off-policy（离线策略） | On-policy（在线策略） |
| 核心机制 | 学习 Q 值表 | 直接学习策略函数 |
| 多进程支持 | 收益很小 | 收益显著（线性加速） |
| 收敛速度 | 慢，需要 500k+ 步 | 快，300k 步通常够 |
| 稳定性 | 一般，容易学会"悬停" | 好，较少陷入局部最优 |
| 适用场景 | 离散动作、样本效率要求高 | 离散/连续动作均可 |
| 推荐训练步数 | 1,000,000 | 4,000,000 |

**结论**：入门推荐直接用 PPO，DQN 作为学习经典算法的参考。

---

## 11. 理解训练输出

### 9.1 DQN 训练输出

```
| rollout/            |          |
|    ep_len_mean      | 605      |
|    ep_rew_mean      | 155      |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 496      |
|    fps              | 2978     |
|    time_elapsed     | 58       |
|    total_timesteps  | 173610   |
| train/              |          |
|    learning_rate    | 0.001    |
|    loss             | 0.366    |
|    n_updates        | 43152    |
```

**rollout（采样统计）**

| 字段 | 含义 | 怎么看 |
|------|------|--------|
| `ep_len_mean` | 最近若干回合的平均步数 | 过高（接近 1000）说明飞船在悬停而不降落 |
| `ep_rew_mean` | 最近若干回合的平均奖励 | 核心指标，超过 200 视为通关 |
| `exploration_rate` | 当前随机探索率（ε） | 训练初期高（随机多），后期降到 0.05 |

**time（时间统计）**

| 字段 | 含义 |
|------|------|
| `episodes` | 已完成的总回合数 |
| `fps` | 每秒执行的环境步数，越高训练越快 |
| `time_elapsed` | 已训练秒数 |
| `total_timesteps` | 已执行的总步数 |

**train（网络训练）**

| 字段 | 含义 | 怎么看 |
|------|------|--------|
| `learning_rate` | 当前学习率 | 固定值则不变 |
| `loss` | Q 网络的预测误差 | 不需要越低越好，波动正常，持续极高才需关注 |
| `n_updates` | 神经网络已更新次数 | 越多说明训练越充分 |

---

### 9.2 PPO 训练输出

```
| rollout/            |          |
|    ep_len_mean      | 312      |
|    ep_rew_mean      | 187      |
| time/               |          |
|    fps              | 8412     |
|    iterations       | 10       |
|    time_elapsed     | 25       |
|    total_timesteps  | 163840   |
| train/              |          |
|    approx_kl        | 0.008    |
|    clip_fraction    | 0.12     |
|    clip_range       | 0.2      |
|    entropy_loss     | -1.21    |
|    explained_var    | 0.94     |
|    learning_rate    | 0.0003   |
|    loss             | 12.3     |
|    n_updates        | 90       |
|    policy_gradient  | -0.031   |
|    value_loss       | 24.6     |
```

**rollout（采样统计）**

| 字段 | 含义 | 怎么看 |
|------|------|--------|
| `ep_len_mean` | 平均回合步数 | 降低说明飞船更快落地 |
| `ep_rew_mean` | 平均回合奖励 | 核心指标，目标 >200 |

**time（时间统计）**

| 字段 | 含义 | 怎么看 |
|------|------|--------|
| `fps` | 每秒步数 | 16 进程下通常比 DQN 高 2~4 倍 |
| `iterations` | 已完成的训练轮次（每轮 = n_steps × n_envs 步） | |
| `total_timesteps` | 已执行总步数 | |

**train（网络训练）**

| 字段 | 含义 | 怎么看 |
|------|------|--------|
| `approx_kl` | 新旧策略的差异程度 | 建议 <0.02，过大说明更新幅度太激进 |
| `clip_fraction` | 被裁剪的样本比例 | 建议 0.1~0.3，过高说明策略变化太快 |
| `clip_range` | PPO 裁剪范围（超参数，固定 0.2） | 控制每次策略更新幅度上限 |
| `entropy_loss` | 策略熵，负数越大说明动作越多样 | 太小（接近 0）表示策略过早收敛 |
| `explained_var` | 价值函数对实际回报的解释度 | 越接近 1 越好，<0.5 说明价值网络没训好 |
| `policy_gradient` | 策略梯度损失 | 负数正常 |
| `value_loss` | 价值网络预测误差 | 训练初期高，随训练下降为正常 |

---

## 12. 理解训练过程

### 训练曲线示意

```
奖励
 200 |                                    ****
 100 |                          **********
   0 |               ***********
-100 |      **********
-200 | ****
     +----+----+----+----+----+----→ 训练步数
     0   20k  40k  60k  80k 100k
```

- **前期（0~30k步）**：智能体在随机探索，奖励很低
- **中期（30k~70k步）**：开始学到有效策略，奖励上升
- **后期（70k~100k步）**：策略趋于稳定，平均奖励超过 200

### 关键超参数说明

| 参数 | 作用 | 建议值 |
|------|------|--------|
| `learning_rate` | 每次更新幅度 | 1e-3 ~ 1e-4 |
| `buffer_size` | 存储经验的容量 | 50000 |
| `batch_size` | 每次学习的样本数 | 64 |
| `exploration_fraction` | 探索阶段占比 | 0.1~0.2 |
| `gamma` | 未来奖励折扣因子 | 0.99 |

---

## 13. 常见问题

**Q: 训练很久但效果不好？**
- 尝试增加 `total_timesteps`（比如 200_000）
- 检查学习率是否太大或太小
- 换用 PPO 算法（通常更稳定）：`from stable_baselines3 import PPO`

**Q: 安装 box2d 报错？**
```bash
# macOS
brew install swig
pip install gymnasium[box2d]

# Linux
sudo apt-get install swig
pip install gymnasium[box2d]
```

**Q: 有没有更好的算法？**
对于 LunarLander，推荐尝试顺序：
```
DQN → PPO → SAC（连续版）
```

**Q: 得分多少算好？**

| 得分 | 评价 |
|------|------|
| < 0 | 很差，基本在坠毁 |
| 0 ~ 100 | 还在学习中 |
| 100 ~ 200 | 有进步，能降落但不稳 |
| > 200 | 成功！官方认定的通关标准 |

---

## 参考资源

- [Gymnasium 官方文档](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
- [Stable-Baselines3 文档](https://stable-baselines3.readthedocs.io/)
- [DQN 原始论文（Mnih et al., 2015）](https://www.nature.com/articles/nature14236)
