"""
LunarLander - Demo SAC（连续动作版）
=====================================
使用 SAC 算法训练 LunarLanderContinuous-v3。

连续版与离散版的区别：
  - 离散版：选择 0/1/2/3 四个动作之一
  - 连续版：输出两个浮点数 [-1, 1]，分别控制主引擎和侧引擎油门

SAC（Soft Actor-Critic）是连续控制的主流算法，
通过最大化熵来鼓励探索，比 DQN 采样效率高，比 PPO 更适合连续动作。

运行方式：
  python demo_sac.py --mode random     # 随机策略演示
  python demo_sac.py --mode train      # 训练 SAC
  python demo_sac.py --mode test       # 测试最佳模型
  python demo_sac.py --mode all        # 依次运行全部

  --no-render        不显示可视化窗口
  --timesteps N      训练总步数（默认 500000）
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

# macOS Chinese font support
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "STHeiti", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ENV_ID          = "LunarLanderContinuous-v3"
MODEL_PATH      = "final_sac"
BEST_MODEL_PATH = "best_sac"


# ─────────────────────────────────────────────
# 1. 随机策略
# ─────────────────────────────────────────────
def run_random(n_episodes=10, render=True):
    print("\n" + "="*50)
    print("  随机策略演示（连续动作版）")
    print("="*50)

    render_mode = "human" if render else None
    env = gym.make(ENV_ID, render_mode=render_mode)

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset(seed=ep)
        total_reward = 0
        steps = 0
        while True:
            action = env.action_space.sample()  # 随机连续动作
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break
        status = "坠毁" if total_reward < 0 else "降落"
        print(f"  回合 {ep:2d}: {steps:4d} 步 | 得分 {total_reward:7.1f} | {status}")

    env.close()
    print("\n随机策略平均得分通常在 -200 ~ 0 之间，这是正常的。\n")


# ─────────────────────────────────────────────
# 2. 回调：保存最佳模型 + 记录奖励
# ─────────────────────────────────────────────
class BestModelCallback(BaseCallback):
    """每隔 eval_freq 步评估一次，自动保存最高分模型到 save_path.zip"""

    def __init__(self, eval_env, save_path, eval_freq=10_000, n_eval_episodes=10):
        super().__init__()
        self.eval_env         = eval_env
        self.save_path        = save_path
        self.eval_freq        = eval_freq
        self.n_eval_episodes  = n_eval_episodes
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            mean_reward, _ = evaluate_policy(
                self.model, self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True,
            )
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(self.save_path)
                print(f"  [Best] 新最高均分 {mean_reward:.1f} -> 已保存到 {self.save_path}.zip")
        return True


class RewardLogCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self._current_reward = 0.0

    def _on_step(self) -> bool:
        self._current_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.episode_rewards.append(self._current_reward)
            self._current_reward = 0.0
        return True

    def plot(self):
        if not self.episode_rewards:
            return
        rewards  = np.array(self.episode_rewards)
        window   = min(20, len(rewards))
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")

        plt.figure(figsize=(10, 4))
        plt.plot(rewards, alpha=0.3, color="mediumseagreen", label="每回合奖励")
        plt.plot(
            range(window - 1, len(rewards)),
            smoothed,
            color="mediumseagreen", linewidth=2,
            label=f"滑动均值（窗口={window}）",
        )
        plt.axhline(200, color="green", linestyle="--", linewidth=1, label="通关线 (200)")
        plt.xlabel("回合数")
        plt.ylabel("累计奖励")
        plt.title("LunarLanderContinuous SAC 训练曲线")
        plt.legend()
        plt.tight_layout()
        plt.savefig("training_curve_sac.png", dpi=150)
        print("  训练曲线已保存到 training_curve_sac.png")
        plt.show()


# ─────────────────────────────────────────────
# 3. 训练
# ─────────────────────────────────────────────
def run_train(total_timesteps=500_000):
    print("\n" + "="*50)
    print("  SAC 训练（连续动作版）")
    print("="*50)
    print(f"  环境: {ENV_ID}")
    print(f"  总步数: {total_timesteps:,}")
    print(f"  最佳模型保存到: {BEST_MODEL_PATH}.zip\n")

    env      = gym.make(ENV_ID)
    eval_env = gym.make(ENV_ID)

    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=200_000,      # 经验回放池
        learning_starts=1_000,    # 先收集 1000 步再开始学习
        batch_size=256,
        tau=0.005,                # 目标网络软更新系数
        gamma=0.99,
        train_freq=1,             # 每步都训练
        gradient_steps=1,
        ent_coef="auto",          # 自动调整熵系数（SAC 核心特性）
        verbose=1,
    )

    best_cb   = BestModelCallback(eval_env, save_path=BEST_MODEL_PATH, eval_freq=10_000)
    reward_cb = RewardLogCallback()
    model.learn(total_timesteps=total_timesteps, callback=[best_cb, reward_cb])
    model.save(MODEL_PATH)

    print(f"\n  最终模型已保存到 {MODEL_PATH}.zip")
    print(f"  最佳模型已保存到 {BEST_MODEL_PATH}.zip")

    if reward_cb.episode_rewards:
        last50 = reward_cb.episode_rewards[-50:]
        print(f"  最后 50 回合平均得分: {np.mean(last50):.1f}")
        print(f"  最高得分: {max(reward_cb.episode_rewards):.1f}")

    reward_cb.plot()
    eval_env.close()
    env.close()


# ─────────────────────────────────────────────
# 4. 测试
# ─────────────────────────────────────────────
def run_test(n_episodes=10, render=True):
    print("\n" + "="*50)
    print("  测试 SAC 最佳模型（连续动作版）")
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

    model = SAC.load(load_path)
    render_mode = "human" if render else None
    env = gym.make(ENV_ID, render_mode=render_mode)

    all_rewards = []
    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset(seed=ep * 10)
        total_reward = 0
        steps = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break
        all_rewards.append(total_reward)
        status = "成功" if total_reward >= 200 else ("降落" if total_reward > 0 else "坠毁")
        print(f"  回合 {ep:2d}: {steps:4d} 步 | 得分 {total_reward:7.1f} | {status}")

    env.close()
    avg = np.mean(all_rewards)
    print(f"\n  平均得分: {avg:.1f}")
    if avg >= 200:
        print("  达到通关标准（>=200），训练成功！")
    else:
        print("  未达通关标准，可以尝试增加训练步数。")


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="LunarLander Demo SAC（连续动作版）")
    parser.add_argument(
        "--mode",
        choices=["random", "train", "test", "all"],
        default="all",
    )
    parser.add_argument("--no-render", action="store_true", help="不显示可视化窗口")
    parser.add_argument("--timesteps", type=int, default=500_000, help="训练总步数（默认: 500000）")
    args = parser.parse_args()

    render = not args.no_render

    if args.mode == "random":
        run_random(render=render)
    elif args.mode == "train":
        run_train(total_timesteps=args.timesteps)
    elif args.mode == "test":
        run_test(render=render)
    elif args.mode == "all":
        run_random(render=render)
        run_train(total_timesteps=args.timesteps)
        run_test(render=render)


if __name__ == "__main__":
    main()
