import os
# Pygame DLL 로드 문제 해결을 위해 가장 먼저 임포트
import pygame

# Gymnasium의 Box2D 환경 로드 시 CarRacing 의존성 오류 무시
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
    [Target]  Customized BipedalWalker-v3
    
    기존 강화학습 환경(BipedalWalker-v3)을 산업 현장의 요구사항에 맞춰 재설계한 래퍼 클래스입니다.
    단순한 보행을 넘어, 화물 적재(Heavy)나 미끄러운 바닥(Slippery)과 같은 극한 환경에서도
    안정적으로 작동하는 제어기를 개발하는 것이 목표입니다.
    """
    
    def __init__(self, env: gym.Env, mode: str = "normal"):
        super().__init__(env)
        self.mode = mode
        
        # 1. State Space Engineering (Feature Selection)
        # ----------------------------------------------------------------
        # [Problem] 기존 24차원 상태 공간에는 Lidar 센서의 노이즈가 포함되어 있어 학습의 수렴을 저해함.
        # [Solution] 자세 제어(Balancing)에 필수적인 핵심 정보 10가지만 선별하여 차원의 저주(Curse of Dimensionality)를 해결.
        # [Effect]  Sim-to-Real 적용 시 고가의 Lidar 센서 의존도를 낮추고 데이터 처리 대역폭을 최적화함.
        #
        # - Hull (4): Angle, Angular Velocity, Velocity X, Velocity Y
        # - Leg 1 (4): Hip Angle/Vel, Knee Angle/Vel
        # - Leg 2 (4): Hip Angle/Vel, Knee Angle/Vel
        # - Lidar (2): Forward Obstacle Detection (Minimal)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(14,), 
            dtype=np.float32
        )
        
        # 2. Action Space Engineering (Discretization)
        # ----------------------------------------------------------------
        # [Problem] 연속적인 힘(Continuous Torque) 제어는 초기 탐색 공간이 방대하여 학습이 불안정하고, 
        #           결과 행동이 부자연스러워(Jittering) 실제 로봇 모터에 무리를 줄 수 있음.
        # [Solution] 로봇 공학의 Motion Primitives 개념을 도입하여 의미 있는 9가지 이산 동작으로 매핑.
        # [Effect]  학습 안정성(Stability) 향상 및 동작의 해석 가능성(Explainability) 확보.
        self.action_space = spaces.Discrete(9)
        
        # Action Primitive Map (Torque Vector: [Hip1, Knee1, Hip2, Knee2])
        self.action_map = {
            0: np.array([0.0, 0.0, 0.0, 0.0]),      # STAY: 대기/안정화
            1: np.array([1.0, 1.0, -1.0, -1.0]),    # FORWARD_BIG: 고속 전진 (큰 보폭)
            2: np.array([0.5, 0.5, -0.5, -0.5]),    # FORWARD_SMALL: 정밀 전진 (작은 보폭)
            3: np.array([0.8, -0.5, 0.0, 0.0]),     # LEG_LIFT_LEFT: 좌측 다리 들어올리기 (장애물/계단)
            4: np.array([0.0, 0.0, 0.8, -0.5]),     # LEG_LIFT_RIGHT: 우측 다리 들어올리기
            5: np.array([0.0, 0.6, 0.0, 0.0]),      # KNEE_BEND_LEFT: 충격 흡수 및 자세 제어
            6: np.array([0.0, 0.0, 0.0, 0.6]),      # KNEE_BEND_RIGHT
            7: np.array([0.7, -0.3, 0.7, -0.3]),    # HIP_EXTEND: 골반 확장 (추진력 확보)
            8: np.array([-0.3, 0.7, -0.3, 0.7]),    # CROUCH: 자세 낮추기 (무게중심 안정화)
        }
        
        self.total_energy_used = 0.0
        
    def _extract_features(self, observation: np.ndarray) -> np.ndarray:
        """High-dimensional Raw Data -> Semantic Features (14-dim)"""
        hull_info = observation[0:4]          # Core Body State
        leg1_info = observation[4:8]          # Leg 1 (Hip/Knee Angle & Vel)
        leg2_info = observation[9:13]         # Leg 2 (Hip/Knee Angle & Vel)
        lidar_front = observation[14:16]      # Exteroception (외수용감각) - 전방만 주시
        
        return np.concatenate([hull_info, leg1_info, leg2_info, lidar_front])
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if isinstance(action, np.ndarray):
            action = action.item()
        continuous_action = self.action_map[int(action)]
        
        # 원본 환경에서 reward 받기 (이 보상이 핵심!)
        obs, original_reward, terminated, truncated, info = self.env.step(continuous_action)
        
        reduced_obs = self._extract_features(obs)
        
        # [Reward Engineering V5 - Back to Basics]
        # 핵심 전략: 검증된 원본 보상 함수를 버리지 않고, 최소한의 수정만 가한다.
        # 
        # 원본 BipedalWalker 보상:
        #   - 전진하면 +1 ~ +3 점 (거리에 비례)
        #   - 넘어지면 -100점
        #   - 모터 사용 시 약간의 페널티
        # 
        # 이 보상은 수천 명의 연구자가 검증한 좋은 설계이므로,
        # 이를 기반으로 "걸음걸이 품질"만 약간 조정한다.
        #
        # Formula: R = R_original + R_gait_bonus - R_energy
        
        # === 1. 원본 보상 (핵심) ===
        # 전진, 생존, 넘어짐 페널티 등이 이미 잘 설계되어 있음
        base_reward = original_reward
        
        # === 2. Gait Bonus (걸음걸이 품질) ===
        # 두 다리의 엉덩이 관절 각도가 반대 부호이면 보너스
        # (한 다리는 앞으로, 한 다리는 뒤로 = 자연스러운 보행 자세)
        hip1_angle = reduced_obs[4]  # Leg 1 Hip Angle
        hip2_angle = reduced_obs[8]  # Leg 2 Hip Angle
        
        gait_bonus = 0.0
        if hip1_angle * hip2_angle < -0.01:  # 반대 부호 = 다리가 벌어져 있음
            gait_bonus = 0.3
            
        # === 3. Energy Penalty (효율성) ===
        # 매우 가벼운 페널티로, 불필요한 경련 방지
        energy_penalty = np.sum(np.abs(continuous_action)) * 0.002
        self.total_energy_used += np.sum(np.abs(continuous_action))
        
        # === Total Reward ===
        # 원본 보상이 메인, 나머지는 미세 조정
        custom_reward = base_reward + gait_bonus - energy_penalty
        
        # Logging info
        info.update({
            'original_reward': original_reward,
            'gait_bonus': gait_bonus,
            'energy_used': self.total_energy_used
        })
        
        return reduced_obs, custom_reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        if options is not None and 'mode' in options:
            self.mode = options['mode']
        
        obs, info = self.env.reset(seed=seed)
        self.total_energy_used = 0.0
        
        # 4. Domain Randomization (Sim-to-Real Robustness)
        # ----------------------------------------------------------------
        # 시뮬레이션 환경의 물리 파라미터를 무작위로 변형하여,
        # 모델이 특정 환경에 과적합(Overfitting)되지 않고 범용적인 대응 능력을 갖추도록 함.
        self._apply_domain_randomization()
        
        reduced_obs = self._extract_features(obs)
        info['mode'] = self.mode
        
        return reduced_obs, info
    
    def _apply_domain_randomization(self):
        """환경 모드에 따른 물리 엔진 파라미터(Mass, Friction) 동적 조절"""
        try:
            unwrapped = self.env.unwrapped
            
            if self.mode == "heavy":
                # [Scenario] 화물 적재: 다리 질량 2배 증가
                # Expectation: 더 큰 토크와 신중한 보행 전략이 요구됨.
                for leg in unwrapped.legs:
                    if leg is not None:
                        for fixture in leg.fixtures:
                            fixture.density *= 2.0
                        leg.ResetMassData()
                        
            elif self.mode == "slippery":
                # [Scenario] 빙판길/유막: 지면 마찰계수 0.2배 감소
                # Expectation: 미끄러짐을 방지하기 위해 보폭을 줄이고 무게중심을 낮추는 전략 필요.
                if hasattr(unwrapped, 'world') and unwrapped.world.bodies:
                    for body in unwrapped.world.bodies:
                        if not body.active or body.type == 0:  # Static bodies (Ground)
                            for fixture in body.fixtures:
                                fixture.friction *= 0.2
                
                for leg in unwrapped.legs:
                    if leg is not None:
                        for fixture in leg.fixtures:
                            fixture.friction *= 0.2
                            
        except AttributeError:
            print(f"Warning: Failed to apply domain randomization for mode '{self.mode}'")


def make_custom_walker(mode: str = "normal") -> gym.Env:
    base_env = gym.make("BipedalWalker-v3")
    return CustomWalkerWrapper(base_env, mode=mode)


if __name__ == "__main__":
    # 환경 구현 검증 (Unit Test)
    print("Verifying CustomWalkerWrapper implementation...")
    
    for mode in ["normal", "heavy", "slippery"]:
        print(f"\n[Test Mode: {mode}]")
        env = make_custom_walker(mode=mode)
        
        print(f"Observation Space: {env.observation_space}")
        print(f"Action Space: {env.action_space}")
        
        obs, info = env.reset()
        print(f"Initial State Shape: {obs.shape}")
        
        total_reward = 0
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        print(f"Episode Reward (100 steps): {total_reward:.2f}")
        env.close()
    
    print("\nVerification Complete.")
