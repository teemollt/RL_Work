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

from PIL import Image, ImageDraw, ImageFont

def add_text_to_frame(frame, text):
    """프레임에 텍스트(Step 수)를 덮어씌움"""
    pil_img = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_img)
    
    # 폰트 로드 (윈도우 기본 폰트 시도)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()
        
    position = (15, 15)
    text_color = (255, 255, 255) # 흰색
    outline_color = (0, 0, 0)    # 검은색 테두리
    
    x, y = position
    # 글씨 테두리 (가독성 향상)
    draw.text((x-1, y-1), text, font=font, fill=outline_color)
    draw.text((x+1, y-1), text, font=font, fill=outline_color)
    draw.text((x-1, y+1), text, font=font, fill=outline_color)
    draw.text((x+1, y+1), text, font=font, fill=outline_color)
    
    draw.text(position, text, font=font, fill=text_color)
    
    return np.array(pil_img)

def create_progression_video(checkpoint_dir: str, env_mode: str = "normal", output_file: str = "progression.gif"):
    """
    저장된 체크포인트들을 순서대로 불러와서 학습 과정 변화(Progression)를 하나의 영상으로 합침.
    예: 0% 시점 (비틀거림) -> 50% 시점 (걷기 시작) -> 100% 시점 (잘 걸음)
    """
    if not os.path.exists(checkpoint_dir):
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        return

    # 체크포인트 파일 찾기 및 정렬 (Steps 기준)
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".zip")]
    
    if not checkpoints:
        print("No checkpoints found.")
        return
        
    # 파일명에서 step 숫자 추출하여 정렬 (예: rl_model_20000_steps.zip 또는 ppo_walker_0_steps.zip)
    import re
    def get_steps(filename):
        match = re.search(r'(\d+)_steps', filename)
        if match:
            return int(match.group(1))
        return 0
        
    checkpoints.sort(key=get_steps)
    
    # [Video Editing] 사용자 요청: 0 Step과 10만 단위 Step만 선택
    # 너무 많은 체크포인트가 있으면 영상이 길어지므로 핵심 구간만 필터링합니다.
    filtered_checkpoints = []
    for cp in checkpoints:
        step = get_steps(cp)
        # 0 step 또는 100,000의 배수인 경우만 포함
        if step == 0 or step % 100000 == 0:
            filtered_checkpoints.append(cp)
    
    # 만약 필터링 결과가 너무 적으면(예: 초기 단계), 그냥 다 보여줌
    if len(filtered_checkpoints) > 1:
        checkpoints = filtered_checkpoints
        print(f"Filtered checkpoints for progression: {len(checkpoints)} frames selected.")
    
    all_frames = []
    
    print(f"\n[Progression Video] Making video from {len(checkpoints)} checkpoints...")
    
    # 환경 설정 (한 번만 생성)
    env = make_custom_walker(mode=env_mode)
    env.unwrapped.render_mode = "rgb_array"
    
    import imageio.v2 as imageio_v2 # 호환성
    
    for ckpt in checkpoints:
        ckpt_path = os.path.join(checkpoint_dir, ckpt)
        step_count = ckpt.split('_')[1]
        print(f"  Processing checkpoint: {step_count} steps")
        
        model = PPO.load(ckpt_path)
        
        # 짧게(300 step) 녹화하여 이어붙이기
        frames, reward = record_episode(model, env, max_steps=300)
        
        # 각 프레임에 현재 Step 수 텍스트 추가
        text = f"Training Steps: {step_count}"
        frames = [add_text_to_frame(f, text) for f in frames]
        
        all_frames.extend(frames)
    
    env.close()
    
    imageio_v2.mimsave(output_file, all_frames, duration=1000/45, loop=0)
    print(f"Progression video saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to single model for evaluation")
    parser.add_argument("--checkpoint_dir", type=str, help="Directory containing checkpoints for progression video")
    parser.add_argument("--output", type=str, default="videos", help="Output directory")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # 1. 단일 모델 평가 (기존 기능)
    if args.model:
        print(f"Loading model from: {args.model}")
        model = PPO.load(args.model)
        
        test_modes = ["normal", "heavy", "slippery"]
        
        for mode in test_modes:
            print(f"\nRecording scenario: {mode}")
            
            env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
            env = make_custom_walker(mode=mode)
            env.unwrapped.render_mode = "rgb_array"
            
            frames, reward = record_episode(model, env)
            print(f"  Result: Reward={reward:.2f}, Frames={len(frames)}")
            
            save_path = os.path.join(args.output, f"walker_{mode}.gif")
            imageio.mimsave(save_path, frames, duration=1000/45, loop=0)
            print(f"  Saved to: {save_path}")
            
            env.close()
            
    # 2. 발전 과정 영상 (Progression Video)
    if args.checkpoint_dir:
        save_path = os.path.join(args.output, "training_progression.gif")
        create_progression_video(args.checkpoint_dir, env_mode="normal", output_file=save_path)

if __name__ == "__main__":
    main()
