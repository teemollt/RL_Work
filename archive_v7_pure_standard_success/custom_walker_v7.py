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
    
    [Version] V7 - Pure Standard (순정 회귀)
    
    이 버전은 "튜닝의 역설"을 해결하기 위해, 모든 커스텀 로직을 제거하고
    OpenAI Gym의 표준 BipedalWalker-v3 환경을 그대로 사용합니다.
    
    1. State Space: 24차원 (Full Feature)
       - Lidar, 관절 속도, 접촉 정보 등 모든 정보를 그대로 제공
       
    2. Action Space: Continuous (Box 4)
       - 4개의 모터 토크를 -1.0 ~ 1.0 사이에서 자유롭게 조절
       
    3. Reward: Original
       - BipedalWalker-v3의 표준 보상 함수 사용
    """
    
    def __init__(self, env: gym.Env, mode: str = "normal"):
        super().__init__(env)
        self.mode = mode
        
        # 순정 환경의 Space 그대로 사용
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
        self.total_energy_used = 0.0
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # Action is fast-forwarded directly to the env
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 에너지 사용량 로깅 (분석용)
        energy = np.sum(np.abs(action))
        self.total_energy_used += energy
        
        info.update({
            'energy_used': self.total_energy_used,
            'original_reward': reward
        })
        
        return obs, reward, terminated, truncated, info
    
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
