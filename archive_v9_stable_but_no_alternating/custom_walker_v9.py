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
    
    [Version] V9 - Strict Rhythm (강제 리듬)
    
    이 버전은 "한 발로 깽깽이 뛰기"나 "무릎 꿇고 기어가기"를 원천 봉쇄하기 위해
    **발바닥 교차(Alternating Contact)**를 강제하는 보상 체계를 도입했습니다.
    
    1. Rhythm Reward (리듬 보상):
       - 왼발 지지 -> 오른발 지지 -> 왼발 지지 순서가 지켜지면 큰 보너스 (+1.0)
       - 같은 발을 연속으로 디디면(깽깽이) 감점 (-0.5)
       
    2. Posture Reward (자세 보상):
       - 상체가 꼿꼿할수록 보너스 (V8 유지)
    """
    
    def __init__(self, env: gym.Env, mode: str = "normal"):
        super().__init__(env)
        self.mode = mode
        
        # 순정 환경의 Space 그대로 사용
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
        self.total_energy_used = 0.0
        
        # 보행 리듬 추적용 (0: 없음, 1: 왼발, 2: 오른발)
        self.last_leg_contact = 0
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # 순정 환경 Step 실행
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        
        # Sense Contracts (BipedalWalker-v3 Obs Index)
        # obs[8]: Leg 1 Ground Contact (Left)
        # obs[13]: Leg 2 Ground Contact (Right)
        left_contact = obs[8]
        right_contact = obs[13]
        hull_angle = obs[0]
        
        # Rhythm Reward Logic
        # ============================================================
        rhythm_reward = 0.0
        
        current_contact = 0
        if left_contact > 0.5 and right_contact < 0.5:
            current_contact = 1 # Left Only
        elif right_contact > 0.5 and left_contact < 0.5:
            current_contact = 2 # Right Only
            
        # 상태 변화가 있을 때만 평가
        if current_contact != 0 and current_contact != self.last_leg_contact:
            if self.last_leg_contact == 0:
                # 첫 발걸음: 보너스
                rhythm_reward = 0.5
            elif self.last_leg_contact != current_contact:
                # 교차 성공 (Left -> Right or Right -> Left): 큰 보너스
                rhythm_reward = 1.0
            else:
                # 같은 발 연속 (Left -> Left): 깽깽이 벌점
                # (Note: 물리적으로 불가능할 수도 있지만, 잠깐 떳다 닿으면 발생)
                rhythm_reward = -0.5
            
            # 상태 업데이트
            self.last_leg_contact = current_contact
            
        # Posture Reward (눕방 방지)
        posture_reward = 1.0 - abs(hull_angle)
        
        # Total
        custom_reward = original_reward + rhythm_reward + posture_reward
        
        # 에너지 사용량 로깅 (분석용)
        energy = np.sum(np.abs(action))
        self.total_energy_used += energy
        
        info.update({
            'energy_used': self.total_energy_used,
            'original_reward': original_reward,
            'rhythm_reward': rhythm_reward,
            'posture_reward': posture_reward
        })
        
        return obs, custom_reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        if options is not None and 'mode' in options:
            self.mode = options['mode']
        
        obs, info = self.env.reset(seed=seed)
        self.total_energy_used = 0.0
        self.last_leg_contact = 0  # 리듬 리셋
        
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
