import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import argparse
from custom_walker import make_custom_walker

class LoggingCallback(BaseCallback):
    """
    학습 과정의 메트릭(Reward, Length, Energy)을 기록하기 위한 커스텀 콜백.
    Tensorboard 외에 별도의 CSV 분석을 위해 데이터를 수집함.
    """
    def __init__(self, mode: str, seed: int, verbose: int = 0):
        super().__init__(verbose)
        self.mode = mode
        self.seed = seed
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_energies = []
        
    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                info = self.locals["infos"][i]
                
                if "episode" in info:
                    ep_reward = info["episode"]["r"]
                    ep_length = info["episode"]["l"]
                    energy = info.get("energy_used", 0)
                    
                    self.episode_rewards.append(ep_reward)
                    self.episode_lengths.append(ep_length)
                    self.episode_energies.append(energy)
                    
                    if self.verbose > 0:
                        print(f"[{self.mode} | Seed {self.seed}] "
                              f"Ep {len(self.episode_rewards)}: "
                              f"Reward={ep_reward:.2f}, Length={ep_length}")
        return True
    
    def get_logs(self) -> dict:
        return {
            "mode": self.mode,
            "seed": self.seed,
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "episode_energies": self.episode_energies,
        }

def train_agent(
    mode: str = "normal", 
    seed: int = 42, 
    total_timesteps: int = 100000, 
    save_dir: str = "models", 
    log_dir: str = "logs", 
    verbose: int = 1
) -> tuple:
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 환경 초기화 및 Monitor 래퍼 적용
    env = make_custom_walker(mode=mode)
    env = Monitor(env, filename=None)
    
    # 재현성(Reproducibility) 확보를 위한 시드 고정
    env.reset(seed=seed)
    np.random.seed(seed)
    
    if verbose >= 1:
        print(f"\n{'='*60}")
        print(f"Training Started - Mode: {mode}, Seed: {seed}")
        print(f"Total Timesteps: {total_timesteps}")
        print(f"{'='*60}\n")
    
    # PPO 알고리즘 설정
    # Discrete Action Space에서의 안정성과 Sample Efficiency를 고려하여 PPO 선정.
    # Hyperparameters는 선행 연구(BipedalWalker Benchmark)를 참고하여 튜닝함.
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,        # On-policy 특성을 고려하여 충분한 롤아웃 확보
        batch_size=64,
        n_epochs=10,
        gamma=0.99,          # 장기적인 보상 고려
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        verbose=verbose,
        seed=seed,
        tensorboard_log=log_dir
    )
    
    callback = LoggingCallback(mode=mode, seed=seed, verbose=verbose)
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )
    
    # 학습 모델 저장
    model_path = os.path.join(save_dir, f"ppo_walker_{mode}_seed{seed}")
    model.save(model_path)
    
    print(f"Model saved at: {model_path}")
    env.close()
    
    return model, callback

def evaluate_agent(model_path: str, mode: str, seed: int, num_episodes: int = 10, verbose: int = 1) -> dict:
    """학습된 모델의 성능 평가 (Inference)"""
    model = PPO.load(model_path)
    
    env = make_custom_walker(mode=mode)
    env.reset(seed=seed)
    
    rewards = []
    lengths = []
    energies = []
    
    print(f"\nEvaluating in '{mode}' mode...")
    
    for i in range(num_episodes):
        obs, info = env.reset()
        ep_reward = 0
        ep_len = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_len += 1
            done = terminated or truncated
        
        rewards.append(ep_reward)
        lengths.append(ep_len)
        energies.append(info.get("energy_used", 0))
        
        if verbose >= 1:
            print(f"  Eval Ep {i+1}: Reward={ep_reward:.2f}")
    
    env.close()
    
    return {
        "mode": mode,
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "mean_length": np.mean(lengths),
        "mean_energy": np.mean(energies),
    }

def main():
    parser = argparse.ArgumentParser(description="Robust Bipedal Walker Training Script")
    parser.add_argument("--mode", type=str, default="normal", help="Training environment mode")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 100], help="Random seeds for reproducibility")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--eval", action="store_true", help="Run evaluation after training")
    
    args = parser.parse_args()
    
    all_results = []
    
    # Multi-seed Training Loop
    # 통계적 유의성을 확보하기 위해 다수의 시드에서 실험 수행
    for seed in args.seeds:
        print(f"\nProcessing Seed {seed}...")
        
        model, callback = train_agent(
            mode=args.mode,
            seed=seed,
            total_timesteps=args.timesteps
        )
        
        # Training Log 저장
        logs = callback.get_logs()
        for i, reward in enumerate(logs["episode_rewards"]):
            all_results.append({
                "mode": args.mode,
                "seed": seed,
                "episode": i,
                "reward": reward,
                "length": logs["episode_lengths"][i],
                "energy": logs["episode_energies"][i],
                "phase": "training"
            })
        
        # Cross-Validation (Robustness Check)
        if args.eval:
            print(f"\nStarting Cross-Validation for Seed {seed}...")
            for eval_mode in ["normal", "heavy", "slippery"]:
                model_path = f"models/ppo_walker_{args.mode}_seed{seed}"
                res = evaluate_agent(model_path, mode=eval_mode, seed=seed)
                
                all_results.append({
                    "mode": eval_mode,
                    "seed": seed,
                    "episode": -1,
                    "reward": res["mean_reward"],
                    "length": res["mean_length"],
                    "energy": res["mean_energy"],
                    "phase": "evaluation"
                })
    
    # 결과 데이터셋 저장
    df = pd.DataFrame(all_results)
    df.to_csv("walker_results.csv", index=False)
    print("\nAll results saved to walker_results.csv")

if __name__ == "__main__":
    main()
