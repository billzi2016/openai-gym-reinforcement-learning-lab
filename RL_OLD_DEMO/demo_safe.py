"""
LunarLander - Safe PPO
======================
在 PPO 基础上加入安全奖励塑形（Reward Shaping），
通过训练让模型本身学会"不坠毁"，而不是事后打补丁。

安全机制：
  1. 坠毁额外惩罚       (-500，原本只有 -100)
  2. 低空高速惩罚       (接近地面时速度过快扣分)
  3. 大角度惩罚         (倾斜过大持续扣分)
  4. 着陆奖励加成       (双脚稳稳落地额外加分)

运行方式：
  python demo_safe.py --mode train     # 训练
  python demo_safe.py --mode test      # 测试最佳模型（统计坠毁率）
  python demo_safe.py --mode all       # 依次运行

  --no-render        不显示可视化窗口
  --timesteps N      训练总步数（默认 10000000）
  --n-envs N         并行进程数（默认: 自动检测 CPU 核心数）
"""

import argparse
import multiprocessing
import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import Wrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv

# macOS Chinese font support
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "STHeiti", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

MODEL_PATH      = "final_safe"
BEST_MODEL_PATH = "best_safe"


# ─────────────────────────────────────────────
# 1. 安全奖励包装器
# ─────────────────────────────────────────────
class SafetyRewardWrapper(Wrapper):
    """
    在原始奖励基础上叠加安全相关的惩罚/奖励：

    obs 索引:
      0: x       水平位置
      1: y       垂直高度
      2: vx      水平速度
      3: vy      垂直速度（负数=下降）
      4: angle   倾斜角度
      5: ang_vel 角速度
      6: left_leg
      7: right_leg
    """

    # 超参数
    CRASH_EXTRA_PENALTY   = -500.0   # 坠毁额外惩罚（叠加在环境原有的 -100 上）
    LOW_ALT_THRESHOLD     = 0.5      # 低空判定高度
    SAFE_DESCENT_SPEED    = -1.2     # 允许的最大下降速度（绝对值）
    SPEED_PENALTY_COEFF   = -2.0     # 低空超速每步惩罚系数
    ANGLE_THRESHOLD       = 0.3      # 允许的最大倾斜角（约 17°）
    ANGLE_PENALTY_COEFF   = -1.5     # 大角度每步惩罚系数
    SOFT_LANDING_BONUS    = 50.0     # 双脚平稳着陆额外奖励
    LANDED_FIRE_PENALTY   = -5.0     # 已着地时点火惩罚（抑制振荡）

    def __init__(self, env):
        super().__init__(env)
        self._prev_legs = (0, 0)
        self._landed    = False      # 双脚是否已稳定着地

    def step(self, action):
        # ── 动作层：已着地则强制静止，彻底消除振荡 ──
        if self._landed:
            action = 0

        obs, reward, terminated, truncated, info = self.env.step(action)
        x, y, vx, vy, angle, ang_vel, left_leg, right_leg = obs

        # ── 1. 坠毁额外惩罚 ──────────────────────────
        if terminated and not (left_leg or right_leg):
            reward += self.CRASH_EXTRA_PENALTY

        # ── 2. 低空高速惩罚 ──────────────────────────
        if y < self.LOW_ALT_THRESHOLD and vy < -self.SAFE_DESCENT_SPEED:
            excess = abs(vy) - self.SAFE_DESCENT_SPEED
            reward += self.SPEED_PENALTY_COEFF * excess

        # ── 3. 大角度惩罚 ────────────────────────────
        if abs(angle) > self.ANGLE_THRESHOLD:
            excess = abs(angle) - self.ANGLE_THRESHOLD
            reward += self.ANGLE_PENALTY_COEFF * excess

        # ── 4. 平稳着陆奖励 ──────────────────────────
        prev_l, prev_r = self._prev_legs
        if (left_leg and right_leg) and not (prev_l and prev_r):
            if abs(vy) < 0.5 and abs(angle) < 0.15:
                reward += self.SOFT_LANDING_BONUS
            self._landed = True      # 标记为已着地

        # ── 5. 着地后点火惩罚（训练模型学会"停下来"）──
        if self._landed and action != 0:
            reward += self.LANDED_FIRE_PENALTY

        self._prev_legs = (left_leg, right_leg)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._prev_legs = (0, 0)
        self._landed    = False
        return self.env.reset(**kwargs)


def make_safe_env():
    env = gym.make("LunarLander-v3")
    return SafetyRewardWrapper(env)


# ─────────────────────────────────────────────
# 2. 回调
# ─────────────────────────────────────────────
class BestModelCallback(BaseCallback):
    """
    评估标准：零坠毁优先，其次才是平均分。
    只有在坠毁次数 <= best 时才考虑更新，坠毁次数相同时才比较平均分。
    """
    def __init__(self, eval_env, save_path, eval_freq=20_000, n_eval_episodes=50):
        super().__init__()
        self.eval_env         = eval_env
        self.save_path        = save_path
        self.eval_freq        = eval_freq
        self.n_eval_episodes  = n_eval_episodes
        self.best_crashes     = np.inf   # 主指标：坠毁次数越少越好
        self.best_mean_reward = -np.inf  # 次指标：平均分越高越好

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # 用原始环境评估（不含安全惩罚），看真实表现
            raw_env = gym.make("LunarLander-v3")
            crashes = 0
            rewards = []
            for seed in range(self.n_eval_episodes):
                obs, _ = raw_env.reset(seed=seed)
                ep_reward = 0
                while True:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, r, terminated, truncated, _ = raw_env.step(action)
                    ep_reward += r
                    if terminated or truncated:
                        if terminated and not (obs[6] or obs[7]):
                            crashes += 1
                        break
                rewards.append(ep_reward)
            raw_env.close()

            mean_reward = np.mean(rewards)
            is_better = (
                crashes < self.best_crashes or
                (crashes == self.best_crashes and mean_reward > self.best_mean_reward)
            )
            if is_better:
                self.best_crashes     = crashes
                self.best_mean_reward = mean_reward
                self.model.save(self.save_path)
                print(f"  [Best] 坠毁 {crashes}/{self.n_eval_episodes} 次 | "
                      f"均分 {mean_reward:.1f} -> 已保存到 {self.save_path}.zip")
            else:
                print(f"  [Eval] 坠毁 {crashes}/{self.n_eval_episodes} 次 | 均分 {mean_reward:.1f}")
        return True


class RewardLogCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self._buf = {}

    def _on_step(self) -> bool:
        for i, (r, d) in enumerate(zip(
            self.locals.get("rewards", []),
            self.locals.get("dones",   [])
        )):
            self._buf[i] = self._buf.get(i, 0.0) + r
            if d:
                self.episode_rewards.append(self._buf[i])
                self._buf[i] = 0.0
        return True

    def plot(self):
        if not self.episode_rewards:
            return
        rewards  = np.array(self.episode_rewards)
        window   = min(30, len(rewards))
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")

        plt.figure(figsize=(10, 4))
        plt.plot(rewards, alpha=0.2, color="crimson", label="每回合奖励")
        plt.plot(
            range(window - 1, len(rewards)),
            smoothed,
            color="crimson", linewidth=2,
            label=f"滑动均值（窗口={window}）",
        )
        plt.axhline(200, color="green", linestyle="--", linewidth=1, label="通关线 (200)")
        plt.xlabel("回合数")
        plt.ylabel("累计奖励（含安全惩罚）")
        plt.title("LunarLander Safe PPO 训练曲线")
        plt.legend()
        plt.tight_layout()
        plt.savefig("training_curve_safe.png", dpi=150)
        print("  训练曲线已保存到 training_curve_safe.png")
        plt.show()


# ─────────────────────────────────────────────
# 3. 训练
# ─────────────────────────────────────────────
def run_train(total_timesteps=100_000_000, n_envs=None):
    if n_envs is None:
        n_envs = multiprocessing.cpu_count()

    print("\n" + "="*50)
    print(f"  Safe PPO 训练（{n_envs} 进程）")
    print("="*50)
    print(f"  总步数: {total_timesteps:,}")
    print(f"  实际采样量: {total_timesteps * n_envs:,}")
    print(f"  最佳模型保存到: {BEST_MODEL_PATH}.zip\n")

    env = make_vec_env(make_safe_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    eval_env = make_safe_env()

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=2e-4,       # 比普通 PPO 更小，更稳
        n_steps=2048,
        batch_size=512,
        n_epochs=10,
        gamma=0.999,
        gae_lambda=0.98,
        ent_coef=0.005,           # 更小的熵，减少随机探索，追求稳定
        clip_range=0.15,          # 更小的裁剪范围，每步更新更保守
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
    )

    best_cb   = BestModelCallback(eval_env, save_path=BEST_MODEL_PATH, eval_freq=20_000)
    reward_cb = RewardLogCallback()
    model.learn(total_timesteps=total_timesteps, callback=[best_cb, reward_cb])
    model.save(MODEL_PATH)

    print(f"\n  最终模型已保存到 {MODEL_PATH}.zip")
    print(f"  最佳模型已保存到 {BEST_MODEL_PATH}.zip")

    if reward_cb.episode_rewards:
        last100 = reward_cb.episode_rewards[-100:]
        print(f"  最后 100 回合平均得分: {np.mean(last100):.1f}")
        print(f"  最高得分: {max(reward_cb.episode_rewards):.1f}")

    reward_cb.plot()
    eval_env.close()
    env.close()


# ─────────────────────────────────────────────
# 4. 测试（统计坠毁率）
# ─────────────────────────────────────────────
def run_test(n_episodes=50, render=True):
    print("\n" + "="*50)
    print("  测试 Safe PPO 最佳模型")
    print("="*50)

    best_model_file = f"{BEST_MODEL_PATH}.zip"
    fallback_file   = f"{MODEL_PATH}.zip"

    if os.path.exists(best_model_file):
        load_path = BEST_MODEL_PATH
        print(f"  加载最佳模型: {best_model_file}")
    elif os.path.exists(fallback_file):
        load_path = MODEL_PATH
        print(f"  未找到最佳模型，加载最终模型: {fallback_file}")
    else:
        print("  找不到模型文件，请先运行训练（--mode train）")
        return

    model = PPO.load(load_path)
    render_mode = "human" if render else None
    # 测试用原始环境（不加安全惩罚），看真实得分
    env = gym.make("LunarLander-v3", render_mode=render_mode)

    all_rewards = []
    crashes     = 0

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset(seed=ep * 7)
        total_reward = 0
        steps = 0
        crashed = False

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                # 判断是否坠毁：回合结束但双脚未着地
                if terminated and not (obs[6] or obs[7]):
                    crashed = True
                    crashes += 1
                break

        all_rewards.append(total_reward)
        status = "坠毁" if crashed else ("成功" if total_reward >= 200 else "降落")
        print(f"  回合 {ep:2d}: {steps:4d} 步 | 得分 {total_reward:7.1f} | {status}")

    env.close()

    avg         = np.mean(all_rewards)
    crash_rate  = crashes / n_episodes * 100
    success_rate = sum(r >= 200 for r in all_rewards) / n_episodes * 100

    print(f"\n  {'='*40}")
    print(f"  测试回合数:  {n_episodes}")
    print(f"  平均得分:    {avg:.1f}")
    print(f"  成功率:      {success_rate:.0f}%  (>=200分)")
    print(f"  坠毁率:      {crash_rate:.0f}%")
    print(f"  {'='*40}")


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="LunarLander Safe PPO")
    parser.add_argument(
        "--mode",
        choices=["train", "test", "all"],
        default="all",
    )
    parser.add_argument("--no-render",  action="store_true", help="不显示可视化窗口")
    parser.add_argument("--timesteps",  type=int, default=100_000_000, help="训练总步数（默认: 100000000）")
    parser.add_argument("--n-envs",     type=int, default=None,       help="并行进程数（默认: 自动检测 CPU 核心数）")
    parser.add_argument("--n-episodes", type=int, default=50,         help="测试回合数（默认: 50）")
    args = parser.parse_args()

    render = not args.no_render

    if args.mode == "train":
        run_train(total_timesteps=args.timesteps, n_envs=args.n_envs)
    elif args.mode == "test":
        run_test(n_episodes=args.n_episodes, render=render)
    elif args.mode == "all":
        run_train(total_timesteps=args.timesteps, n_envs=args.n_envs)
        run_test(n_episodes=args.n_episodes, render=render)


if __name__ == "__main__":
    main()
