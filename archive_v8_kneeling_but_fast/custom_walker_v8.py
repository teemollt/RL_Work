import os
import pygame
import sys
from unittest.mock import MagicMock
sys.modules["gymnasium.envs.box2d.car_racing"] = MagicMock()

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple

class CustomWalkerWrapper(gym.Wrapper):
    """
    [Project] 산업용 이족 보행 로봇을 위한 강건한 제어 시스템 (Robust Control System for Industrial Bipedal Robots)
    
    [Version] V8 - Natural Gait (자연스러운 보행)
    
    이 버전은 V7(순정 V3)의 성공을 바탕으로, "사람 같은 보행 스타일"을 유도하기 위해
    **보상 함수만 미세 조정(Fine-tuning)**한 버전입니다.
    
    물리/행동 공간은 순정 BipedalWalker-v3와 100% 동일하게 유지하여 학습 안정성을 확보하고,
    '자세'와 '보폭'에 대한 보너스 항만 추가했습니다.
    
    1. State/Action Space: Standard (24 dim / Box 4)
    
    2. Reward Shaping (스타일 교정):
       - R_total = R_original + R_posture + R_scissor
       - R_posture: 상체를 꼿꼿이 세우면 보너스 (뒤로 눕기 방지)
       - R_scissor: 양 다리의 각도 차이가 클수록 보너스 (시원시원한 보폭 유도)
    """
    
    def __init__(self, env: gym.Env, mode: str = "normal"):
        super().__init__(env)
        self.mode = mode
        
        # 순정 환경의 Space 그대로 사용
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
        self.total_energy_used = 0.0
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # 순정 환경 Step 실행
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        
        # Style Reward Calculation
        # ============================================================
        hull_angle = obs[0]       # 몸통 각도
        hip1_angle = obs[4]       # 다리1 엉덩이 각도
        hip2_angle = obs[9]       # 다리2 엉덩이 각도
        
        # 1. Posture Reward (자세 교정)
        # 눕방 방지: 몸통이 수직일수록 점수 (최대 +1.0)
        # 기존 로봇은 넘어지지 않으려고 일부러 뒤로 눕는데, 이를 막음
        posture_reward = 1.0 - abs(hull_angle)
        
        # 2. Scissor Reward (가위차기/보폭)
        # 펭귄 걸음 방지: 두 다리가 서로 멀어질수록(교차할수록) 점수
        # 큰 보폭으로 성큼성큼 걷도록 유도 (+0.0 ~ +1.0 수준)
        scissor_reward = abs(hip1_angle - hip2_angle) * 0.5
        
        # Total Reward = 성능(Original) + 스타일(Style)
        # Original Reward가 여전히 메인 엔진(생존/전진) 역할
        custom_reward = original_reward + posture_reward + scissor_reward
        
        # 에너지 사용량 로깅 (분석용)
        energy = np.sum(np.abs(action))
        self.total_energy_used += energy
        
        info.update({
            'energy_used': self.total_energy_used,
            'original_reward': original_reward,
            'posture_reward': posture_reward,
            'scissor_reward': scissor_reward
        })
        
        return obs, custom_reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        if options is not None and 'mode' in options:
            self.mode = options['mode']
        
        obs, info = self.env.reset(seed=seed)
        self.total_energy_used = 0.0
        
        # Domain Randomization (평가용 모드일 때만 적용)
        # Normal 모드일 때는 100% 순정 상태 유지
        if self.mode in ["heavy", "slippery"]:
            self._apply_domain_randomization()
            
        info['mode'] = self.mode
        return obs, info
    
    def _apply_domain_randomization(self):
        try:
            unwrapped = self.env.unwrapped
            if self.mode == "heavy":
                for leg in unwrapped.legs:
                    if leg is not None:
                        for fixture in leg.fixtures:
                            fixture.density *= 2.0
                        leg.ResetMassData()
            elif self.mode == "slippery":
                if hasattr(unwrapped, 'world') and unwrapped.world.bodies:
                    for body in unwrapped.world.bodies:
                        if not body.active or body.type == 0:
                            for fixture in body.fixtures:
                                fixture.friction *= 0.2
                for leg in unwrapped.legs:
                    if leg is not None:
                        for fixture in leg.fixtures:
                            fixture.friction *= 0.2
        except AttributeError:
            pass

def make_custom_walker(mode: str = "normal") -> gym.Env:
    base_env = gym.make("BipedalWalker-v3")
    return CustomWalkerWrapper(base_env, mode=mode)

if __name__ == "__main__":
    print("Verifying CustomWalkerWrapper V7 (Standard)...")
    env = make_custom_walker()
    print(f"Observation Space: {env.observation_space.shape}")
    print(f"Action Space: {env.action_space}")
    env.reset()
    print("Verification Complete.")
