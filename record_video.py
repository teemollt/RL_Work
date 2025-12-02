import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import imageio
import argparse
import os
from custom_walker import make_custom_walker

"""
record_video.py
학습된 에이전트의 정성적 평가(Qualitative Analysis)를 위한 시연 영상 녹화 스크립트.
각 환경 조건에서의 행동 패턴을 시각적으로 검증하기 위해 GIF 포맷으로 저장함.
"""

def record_episode(model, env, max_steps: int = 1000) -> tuple:
    frames = []
    obs, info = env.reset()
    done = False
    ep_len = 0
    total_reward = 0
    
    while not done and ep_len < max_steps:
        # Render current state
        frame = env.render()
        frames.append(frame)
        
        # Inference (Deterministic Policy)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        ep_len += 1
        done = terminated or truncated
    
    return frames, total_reward

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--output", type=str, default="videos", help="Output directory")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Loading model from: {args.model}")
    model = PPO.load(args.model)
    
    # 모든 테스트 시나리오에 대해 녹화 수행
    test_modes = ["normal", "heavy", "slippery"]
    
    for mode in test_modes:
        print(f"\nRecording scenario: {mode}")
        
        # 1. 환경 설정 (물리 파라미터 적용)
        env = make_custom_walker(mode=mode)
        
        # 2. 렌더링을 위한 재설정
        # Gymnasium의 render_mode 설정을 위해 환경을 재구성함
        env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
        env = make_custom_walker(mode=mode)
        env.unwrapped.render_mode = "rgb_array"
        
        # 3. 에피소드 녹화
        frames, reward = record_episode(model, env)
        print(f"  Result: Reward={reward:.2f}, Frames={len(frames)}")
        
        # 4. GIF 저장
        save_path = os.path.join(args.output, f"walker_{mode}.gif")
        imageio.mimsave(save_path, frames, fps=30)
        print(f"  Saved to: {save_path}")
        
        env.close()

if __name__ == "__main__":
    main()
