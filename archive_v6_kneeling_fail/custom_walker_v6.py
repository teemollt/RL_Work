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
    
    [Version] V6 - The Logical Gait (논리적 보행)
    
    이 버전은 이전 실패(V3, V4, Baseline)를 분석하여 근본적인 물리적 제약을 해결한 버전입니다.
    
    1. State Space 확장 (14 -> 22 dim):
       - 로봇이 지형을 볼 수 있도록 Lidar 센서 정보를 2개에서 10개로 대폭 복구했습니다.
       - 이제 로봇은 '장님'이 아니며, 발 밑의 지형을 미리 파악하고 발을 뻗을 수 있습니다.
       
    2. Action Space 최적화 (Soft Dynamics):
       - 기존의 과격한 토크(1.0)를 0.6~0.8 수준으로 낮추어, 급발진으로 인한 넘어짐을 방지합니다.
       - 더 세밀한 균형 잡기를 위해 동작을 정교화했습니다.
       
    3. Reward V6 (Alternating Contact):
       - "걷는다"는 정의를 물리적으로 강제합니다.
       - 한 발이 땅에 있을 때 다른 발은 공중에 있어야만 점수를 줍니다 (XOR 조건).
    """
    
    def __init__(self, env: gym.Env, mode: str = "normal"):
        super().__init__(env)
        self.mode = mode
        
        # 1. State Space Engineering (Vision Restoration)
        # ----------------------------------------------------------------
        # - Hull (4): Angle, Angular Velocity, Velocity X, Velocity Y
        # - Leg 1 (4): Hip Angle/Vel, Knee Angle/Vel
        # - Leg 2 (4): Hip Angle/Vel, Knee Angle/Vel
        # - Lidar (10): 전방 지형 정보 (필수)
        # Total: 22 dimensions
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(22,), 
            dtype=np.float32
        )
        
        # 2. Action Space Engineering (Soft Control)
        # ----------------------------------------------------------------
        # 과도한 토크는 시스템 불안정의 주원인이었습니다.
        # 모터 출력을 80% 수준으로 제한하여 안정성을 확보합니다.
        self.action_space = spaces.Discrete(9)
        
        # Torque Vector: [Hip1, Knee1, Hip2, Knee2]
        self.action_map = {
            0: np.array([0.0, 0.0, 0.0, 0.0]),      # STAY: 안정화
            1: np.array([0.8, 0.8, -0.8, -0.8]),    # FORWARD_MAIN: 주행 (토크 0.8 제한)
            2: np.array([0.4, 0.4, -0.4, -0.4]),    # FORWARD_SOFT: 저속 주행 (정밀)
            3: np.array([0.6, -0.3, 0.0, 0.0]),     # LIFT_LEFT: 왼발 들기 (장애물)
            4: np.array([0.0, 0.0, 0.6, -0.3]),     # LIFT_RIGHT: 오른발 들기
            5: np.array([0.0, 0.4, 0.0, 0.0]),      # BEND_LEFT: 왼 무릎 굽힘 (충격 흡수)
            6: np.array([0.0, 0.0, 0.0, 0.4]),      # BEND_RIGHT: 오른 무릎 굽힘
            7: np.array([0.6, 0.0, 0.6, 0.0]),      # PUSH_HIPS: 엉덩이 밀기 (추진력)
            8: np.array([-0.3, 0.5, -0.3, 0.5]),    # CROUCH: 자세 낮추기
        }
        
        self.total_energy_used = 0.0
        
    def _extract_features(self, observation: np.ndarray) -> np.ndarray:
        """High-dimensional Raw Data -> Semantic Features (22-dim)"""
        hull_info = observation[0:4]          # Core Body State
        leg1_info = observation[4:8]          # Leg 1
        leg2_info = observation[9:13]         # Leg 2
        lidar_info = observation[14:24]       # Lidar (10개로 확장 - 지형 인식용)
        
        return np.concatenate([hull_info, leg1_info, leg2_info, lidar_info])
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if isinstance(action, np.ndarray):
            action = action.item()
        continuous_action = self.action_map[int(action)]
        
        obs, original_reward, terminated, truncated, info = self.env.step(continuous_action)
        reduced_obs = self._extract_features(obs)
        
        # [Reward Engineering V6 - Logical Gait]
        # ==================================================================
        # 목표: "한 발은 지지하고, 한 발은 내딛는" 교차 보행을 강제함.
        # Formula: R = R_forward + R_alternating + R_stability - R_energy
        # ==================================================================
        
        # 1. 상태 추출
        hull_angle = reduced_obs[0]
        forward_velocity = reduced_obs[2]
        
        leg1_contact = obs[8]   # 1.0 if touching ground
        leg2_contact = obs[13]
        
        # 2. Main Reward: Forward Progress
        # 전진해야만 기본 점수를 얻음 (x2.0 가중치)
        forward_reward = forward_velocity * 4.0
        
        # 3. Gait Reward: Alternating Contact (XOR)
        # 중요: 한 발만 땅에 닿아 있을 때 가장 높은 점수를 줌
        # 이는 로봇이 두 발을 동시에 질질 끄는 것을 방지함
        gait_reward = 0.0
        is_single_support = (leg1_contact > 0.5) != (leg2_contact > 0.5)
        
        if is_single_support:
            gait_reward = 2.0  # 한 발 지지 시 보너스
            # 전진 속도가 있을 때 한 발 지지하면 더 큰 보너스 (걷는 중)
            if forward_velocity > 0.1:
                gait_reward += 1.0
        
        # 4. Stability Logic (안정성)
        # 상체가 너무 기울어지면(-0.5 ~ 0.5 rad 범위를 벗어나면) 페널티 폭탄
        stability_penalty = 0.0
        if abs(hull_angle) > 0.4:
            stability_penalty = 5.0
            
        # 5. Energy Efficiency
        energy_penalty = np.sum(np.abs(continuous_action)) * 0.005
        self.total_energy_used += np.sum(np.abs(continuous_action))
        
        # 6. Standstill Penalty (나태함 방지)
        # 앞으로 가지 않으면 가만히 있어도 감점
        standstill_penalty = 0.0
        if forward_velocity < 0.05:
            standstill_penalty = 1.0

        # === Total Calculation ===
        custom_reward = (
            forward_reward 
            + gait_reward 
            - stability_penalty 
            - energy_penalty 
            - standstill_penalty
        )
        
        # Logging info
        info.update({
            'forward_reward': forward_reward,
            'gait_reward': gait_reward,
            'stability_penalty': stability_penalty,
            'energy_used': self.total_energy_used
        })
        
        return reduced_obs, custom_reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        if options is not None and 'mode' in options:
            self.mode = options['mode']
        
        obs, info = self.env.reset(seed=seed)
        self.total_energy_used = 0.0
        
        # Domain Randomization
        self._apply_domain_randomization()
        
        reduced_obs = self._extract_features(obs)
        info['mode'] = self.mode
        
        return reduced_obs, info
    
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
    print("Verifying CustomWalkerWrapper V6...")
    env = make_custom_walker()
    print(f"Observation Space: {env.observation_space.shape}")
    print(f"Action Space: {env.action_space}")
    env.reset()
    print("Verification Complete.")
