"""
LunarLander - Demo PPO
======================
使用 PPO 算法 + 16 进程并行训练。
PPO 是 on-policy 算法，多进程采样收益明显，通常比 DQN 收敛更快。

运行方式：
  python demo_ppo.py --mode random     # 随机策略演示
  python demo_ppo.py --mode train      # 训练 PPO
  python demo_ppo.py --mode test       # 测试最佳模型
  python demo_ppo.py --mode all        # 依次运行全部

  --no-render        不显示可视化窗口
  --timesteps N      训练总步数（默认 300000）
  --n-envs N         并行进程数（默认 16）
"""

import argparse
import os
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

# macOS Chinese font support
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "STHeiti", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

MODEL_PATH      = "final_ppo"
BEST_MODEL_PATH = "best_ppo"


# ─────────────────────────────────────────────
# 随机策略
# ─────────────────────────────────────────────
def run_random(n_episodes=10, render=True):
    print("\n" + "="*50)
    print("  随机策略演示")
    print("="*50)

    render_mode = "human" if render else None
    env = gym.make("LunarLander-v3", render_mode=render_mode)

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset(seed=ep)
        total_reward = 0
        steps = 0
        while True:
            action = env.action_space.sample()
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
# 回调：记录奖励 + 保存最佳模型
# ─────────────────────────────────────────────
class BestModelCallback(BaseCallback):
    """每隔 eval_freq 步评估一次，自动保存最高分模型到 save_path.zip"""

    def __init__(self, eval_env, save_path, eval_freq=10_000, n_eval_episodes=10):
        super().__init__()
        self.eval_env        = eval_env
        self.save_path       = save_path
        self.eval_freq       = eval_freq
        self.n_eval_episodes = n_eval_episodes
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
        self._buf = {}  # env_idx -> 当前回合累计奖励

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [])
        dones   = self.locals.get("dones",   [])
        for i, (r, d) in enumerate(zip(rewards, dones)):
            self._buf[i] = self._buf.get(i, 0.0) + r
            if d:
                self.episode_rewards.append(self._buf[i])
                self._buf[i] = 0.0
        return True

    def plot(self):
        if not self.episode_rewards:
            return
        rewards  = np.array(self.episode_rewards)
        window   = min(20, len(rewards))
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")

        plt.figure(figsize=(10, 4))
        plt.plot(rewards, alpha=0.3, color="darkorange", label="每回合奖励")
        plt.plot(
            range(window - 1, len(rewards)),
            smoothed,
            color="darkorange", linewidth=2,
            label=f"滑动均值（窗口={window}）",
        )
        plt.axhline(200, color="green", linestyle="--", linewidth=1, label="通关线 (200)")
        plt.xlabel("回合数")
        plt.ylabel("累计奖励")
        plt.title("LunarLander PPO 训练曲线（16 进程）")
        plt.legend()
        plt.tight_layout()
        plt.savefig("training_curve_ppo.png", dpi=150)
        print("  训练曲线已保存到 training_curve_ppo.png")
        plt.show()


# ─────────────────────────────────────────────
# 训练
# ─────────────────────────────────────────────
def run_train(total_timesteps=4_000_000, n_envs=None):
    print("\n" + "="*50)
    if n_envs is None:
        n_envs = multiprocessing.cpu_count()
    print(f"  PPO 训练（{n_envs} 进程）")
    print("="*50)
    print(f"  总步数: {total_timesteps:,}")
    print(f"  实际采样量: {total_timesteps * n_envs:,}（进程数 x 步数）")
    print(f"  最佳模型保存到: {BEST_MODEL_PATH}.zip\n")

    # 多进程并行环境
    env = make_vec_env(
        "LunarLander-v3",
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
    )
    eval_env = gym.make("LunarLander-v3")

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=1024,       # 每个进程每轮采集步数，总 batch = n_steps * n_envs
        batch_size=256,
        n_epochs=10,        # 每批数据重复训练次数
        gamma=0.999,
        gae_lambda=0.98,    # GAE 优势估计平滑系数
        ent_coef=0.01,      # 熵正则，防止策略过早收敛
        verbose=1,
    )

    best_cb = BestModelCallback(eval_env, save_path=BEST_MODEL_PATH, eval_freq=10_000)

    reward_cb = RewardLogCallback()
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
# 测试
# ─────────────────────────────────────────────
def run_test(n_episodes=10, render=True):
    print("\n" + "="*50)
    print("  测试 PPO 最佳模型")
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
    env = gym.make("LunarLander-v3", render_mode=render_mode)

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
    parser = argparse.ArgumentParser(description="LunarLander Demo PPO")
    parser.add_argument(
        "--mode",
        choices=["random", "train", "test", "all"],
        default="all",
    )
    parser.add_argument("--no-render", action="store_true", help="不显示可视化窗口")
    parser.add_argument("--timesteps", type=int, default=4_000_000, help="训练总步数（默认: 4000000）")
    parser.add_argument("--n-envs",    type=int, default=None,    help="并行进程数（默认: 自动检测 CPU 核心数）")
    args = parser.parse_args()

    render = not args.no_render

    if args.mode == "random":
        run_random(render=render)
    elif args.mode == "train":
        run_train(total_timesteps=args.timesteps, n_envs=args.n_envs)
    elif args.mode == "test":
        run_test(render=render)
    elif args.mode == "all":
        run_random(render=render)
        run_train(total_timesteps=args.timesteps, n_envs=args.n_envs)
        run_test(render=render)


if __name__ == "__main__":
    main()
