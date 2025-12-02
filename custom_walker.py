import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple

class CustomWalkerWrapper(gym.Wrapper):
    """
    산업용 로봇 제어를 위한 BipedalWalker-v3 커스텀 래퍼.
    
    기존 환경의 문제점인 고차원 상태 공간과 연속 행동 공간의 불안정성을 해결하기 위해
    State Reduction과 Action Discretization을 적용함.
    또한 산업 현장의 요구사항(안정성, 에너지 효율)을 반영하여 보상 함수를 재설계하였음.
    """
    
    def __init__(self, env: gym.Env, mode: str = "normal"):
        super().__init__(env)
        self.mode = mode
        
        # [State Space Reduction] 24-dim -> 10-dim
        # Lidar 센서의 노이즈가 학습 수렴을 저해하는 것으로 판단되어,
        # 자세 제어에 필수적인 Hull 정보와 Joint 정보 위주로 상태 공간을 축소함.
        # 이는 Sim-to-Real 적용 시 센서 비용 절감 효과도 기대할 수 있음.
        #
        # Selected Features:
        # - Hull (4): angle, angular_vel, vel_x, vel_y
        # - Joints (4): hip/knee angle & velocity
        # - Lidar (2): 전방 장애물 감지용 (최소화)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(10,), 
            dtype=np.float32
        )
        
        # [Action Space Discretization] Continuous -> Discrete(9)
        # 연속 제어(Continuous Control)는 초기 탐색 공간이 너무 넓어 학습이 불안정함.
        # 로봇 공학에서 사용되는 Motion Primitives 개념을 도입하여 9가지 동작으로 이산화함.
        # 이를 통해 학습 안정성을 높이고 동작의 해석 가능성(Explainability)을 확보함.
        self.action_space = spaces.Discrete(9)
        
        # Action Mapping Table
        # 각 인덱스는 [hip1, knee1, hip2, knee2]의 토크 벡터에 매핑됨
        self.action_map = {
            0: np.array([0.0, 0.0, 0.0, 0.0]),      # STAY (안전 상태)
            1: np.array([1.0, 1.0, -1.0, -1.0]),    # FORWARD_BIG (고속 보행)
            2: np.array([0.5, 0.5, -0.5, -0.5]),    # FORWARD_SMALL (안정 보행)
            3: np.array([0.8, -0.5, 0.0, 0.0]),     # LEG_LIFT_LEFT (장애물 회피)
            4: np.array([0.0, 0.0, 0.8, -0.5]),     # LEG_LIFT_RIGHT
            5: np.array([0.0, 0.6, 0.0, 0.0]),      # KNEE_BEND_LEFT (자세 제어)
            6: np.array([0.0, 0.0, 0.0, 0.6]),      # KNEE_BEND_RIGHT
            7: np.array([0.7, -0.3, 0.7, -0.3]),    # HIP_EXTEND (추진)
            8: np.array([-0.3, 0.7, -0.3, 0.7]),    # CROUCH (무게중심 하강)
        }
        
        self.total_energy_used = 0.0
        
    def _extract_features(self, observation: np.ndarray) -> np.ndarray:
        """24차원 관측 벡터에서 핵심 특징 10차원 추출"""
        hull_info = observation[0:4]          # Hull state
        joint_info = observation[4:8]         # Joint state
        lidar_front = observation[14:16]      # Frontal Lidar
        
        return np.concatenate([hull_info, joint_info, lidar_front])
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # Discrete index -> Continuous torque 변환
        continuous_action = self.action_map[action]
        
        # 환경 상호작용
        obs, reward, terminated, truncated, info = self.env.step(continuous_action)
        
        # Feature Extraction
        reduced_obs = self._extract_features(obs)
        
        # [Reward Engineering]
        # 기존 보상 함수는 전진 속도에 편향되어 있어 안정성이 떨어짐.
        # 산업용 로봇의 요구사항(Safety First)을 반영하여 안정성 가중치를 높이고
        # 에너지 효율성을 고려한 페널티 항을 추가함.
        #
        # Formula: R = 2.0 * Stability + 1.0 * Forward - Energy_Penalty
        
        # 1. Stability Reward: 직립 자세 유지 및 급격한 회전 방지
        hull_angle = reduced_obs[0]
        angular_velocity = reduced_obs[1]
        stability_reward = 1.0 - abs(hull_angle)
        stability_reward -= abs(angular_velocity) * 0.1
        
        # 2. Forward Reward: 전진 속도 (가중치 축소)
        forward_velocity = reduced_obs[2]
        forward_reward = forward_velocity * 0.1
        
        # 3. Energy Penalty: 토크 사용량 최소화 (배터리 효율)
        energy_penalty = -np.sum(np.abs(continuous_action)) * 0.01
        self.total_energy_used += np.sum(np.abs(continuous_action))
        
        # Total Reward
        custom_reward = 2.0 * stability_reward + 1.0 * forward_reward + energy_penalty
        
        # Logging info
        info.update({
            'energy_used': self.total_energy_used,
            'stability_reward': stability_reward,
            'forward_reward': forward_reward
        })
        
        return reduced_obs, custom_reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        if options is not None and 'mode' in options:
            self.mode = options['mode']
        
        obs, info = self.env.reset(seed=seed)
        self.total_energy_used = 0.0
        
        # [Domain Randomization]
        # 시뮬레이션과 실제 환경의 간극(Sim-to-Real Gap)을 줄이기 위해
        # 물리 파라미터를 랜덤하게 변형하여 강건성(Robustness)을 확보함.
        self._apply_domain_randomization()
        
        reduced_obs = self._extract_features(obs)
        info['mode'] = self.mode
        
        return reduced_obs, info
    
    def _apply_domain_randomization(self):
        """현재 모드에 따른 물리 파라미터 변형 적용"""
        try:
            unwrapped = self.env.unwrapped
            
            if self.mode == "heavy":
                # 시나리오: 무거운 화물 적재 상황
                # 다리 객체의 밀도를 2배 증가시켜 질량 변화에 대한 제어기 성능 검증
                for leg in unwrapped.legs:
                    if leg is not None:
                        for fixture in leg.fixtures:
                            fixture.density *= 2.0
                        leg.ResetMassData()
                        
            elif self.mode == "slippery":
                # 시나리오: 저마찰 지면 (빙판, 젖은 바닥)
                # 지면 및 다리의 마찰 계수를 0.2배로 감소시켜 미끄러짐에 대한 강건성 검증
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
