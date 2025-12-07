# 최종 프로젝트 보고서: 산업용 이족 보행 로봇을 위한 강건한 제어 시스템

**팀원**: [학번] [이름] (GitHub: [Link])

---

## 1. 프로젝트 주제 및 목표

### 1.1 주제
**"안정적이고 인간과 유사한(Human-like) 이족 보행 로봇 제어 시스템 개발"**

### 1.2 목표
OpenAI Gym의 `BipedalWalker-v3` 환경에서 심층 강화학습(Deep Reinforcement Learning)을 활용하여 다음 세 가지 목표를 달성한다.
1.  **High Score**: 공식 해결 기준인 300점 이상 달성.
2.  **Stability**: 다양한 환경(무거운 다리, 미끄러운 지면)에서도 넘어지지 않는 강건성(Robustness) 확보.
3.  **Human-like Gait**: "한 발로 깽깽이(Hopping)"나 "무릎 꿇고 기어가기(Kneeling)" 등의 편법(Reward Hacking)을 배제하고, 두 발을 번갈아 딛는(Alternating Gait) 자연스러운 보행 구현.

---

## 2. 환경 및 데이터셋 설명

### 2.1 환경: BipedalWalker-v3
물리 엔진 Box2D 기반의 이족 보행 시뮬레이션 환경이다.
*   **State Space (24차원)**:
    *   Lidar 센서 (10개): 지형의 높낮이 감지.
    *   관절 각도 및 속도 (Hull Angle, Hip/Knee Angle & Velocity).
    *   지면 접촉 여부 (Ground Contact).
*   **Action Space (4차원 연속 공간)**:
    *   `Hip 1`, `Knee 1`, `Hip 2`, `Knee 2` 모터의 토크 (범위: -1.0 ~ 1.0).
*   **기본 보상 함수 (Original Reward)**:
    *   전진 속도에 비례하여 보상 지급 (최대 130점).
    *   에너지 사용량(Torque)에 따른 페널티.
    *   넘어지면 -100점.

### 2.2 도메인 랜덤화 (Domain Randomization)
로봇의 강건성을 평가하기 위해 세 가지 모드를 구현하였다.
1.  **Normal**: 표준 환경.
2.  **Heavy**: 다리(Leg)의 밀도(Density)를 2.0배 증가시켜, 모터 부하가 큰 상황 시뮬레이션.
3.  **Slippery**: 지면 마찰 계수(Friction)를 0.2배로 낮춰, 빙판길과 유사한 상황 시뮬레이션.

---

## 3. 실험 설계 및 방법 (Methodology)

본 프로젝트는 총 12단계(V1 ~ V12)의 반복적인 실험을 통해 최적의 해결책을 도출하였다.

### 3.1 주요 실패 사례 분석 (Failure Analysis)

| 버전 | 접근 방식 | 결과 | 실패 원인 (Root Cause) |
|---|---|---|---|
| **V1~V2** | 안정성(Stability) 보상 강화 | **제자리 정지** | 전진 보상보다 넘어지는 페널티가 두려워 움직이지 않는 것이 최적 전략(Optimal Policy)이 됨. |
| **V3~V6** | 상태 공간 축소 (State Reduction) | **불안정함** | Lidar 정보를 줄였더니 로봇이 지형을 인식하지 못해 "장님" 상태가 됨. |
| **V7** | **순정(Standard) 회귀** | **성공 (300점)** | 가장 기본적인 설정이 가장 높은 점수를 기록함. 그러나 **보행 스타일이 부자연스러움 (Skiing/Hopping).** |
| **V8~V10** | 리듬/자세 보상 추가 | **부분 성공** | 상체는 펴졌으나, 여전히 두 발을 교차하지 않고 한 발로 미끄러지는 것이 에너지 효율적이라 판단함. |

### 3.2 최종 해결책: V12 - Swing Phase Enforcement (규칙 기반 강제)

단순히 보상(Reward)을 주는 것만으로는 로봇의 "게으른 본성(Local Optima)"을 이길 수 없음을 깨닫고, **환경의 물리적 규칙**을 수정하였다.

1.  **Swing Phase 강제**:
    *   한쪽 발이 땅에 `50 스텝` 이상 연속으로 붙어있으면 에피소드를 **강제로 종료(Termination)**시키고 벌점(-50)을 부여함.
    *   이로 인해 로봇은 살기 위해서라도 발을 들어올려야 함.
2.  **교차 보상 (Alternating Reward)**:
    *   `Left -> Right` 또는 `Right -> Left`로 지지 발이 바뀔 때마다 +3.0점 부여.
3.  **자세 보상 (Posture Reward)**:
    *   상체를 수직으로 유지하면 보너스.

---

## 4. 알고리즘 및 하이퍼파라미터

### 4.1 알고리즘: PPO (Proximal Policy Optimization)
연속적인 행동 공간(Continuous Action Space) 제어에 탁월하고 학습 안정성이 높은 PPO 알고리즘을 사용하였다. (`Stable-Baselines3` 라이브러리 활용)

### 4.2 하이퍼파라미터 (Hyperparameters)
*   **Learning Rate**: 3e-4 (0.0003)
*   **Batch Size**: 64
*   **n_steps**: 2048 (한 번의 업데이트에 사용할 데이터 수)
*   **Gamma (Discount Factor)**: 0.99
*   **GAE Lambda**: 0.95
*   **Total Timesteps**: 500,000 (V12 기준)

---

### 3.2 최종 해결책: V7 + V9 (Transfer Learning & Rhythm Reward)

순정 모델(V7)의 뛰어난 주행 성능을 유지하면서, 스타일을 개선하기 위해 **"전이 학습(Transfer Learning)"**과 **"리듬 보상(Strict Rhythm Reward)"**을 결합한 하이브리드 전략을 채택하였다.

1.  **Transfer Learning (전이 학습)**:
    *   V7(순정)에서 100만 스텝 이상 학습된 모델을 초기 가중치(Pre-trained Weight)로 사용.
    *   이미 걷는 법을 아는 로봇에게 "예쁘게 걷는 법"만 추가로 가르치는 Fine-tuning 전략.
    
2.  **Strict Rhythm Reward (리듬 보상)**:
    *   **교차 보너스**: `Left -> Right` 또는 `Right -> Left`로 발을 바꿀 때마다 **+1.0점**.
    *   **페널티**: 같은 발을 연속으로 디디면(Hopping) **-0.5점**.
    *   **자세 보상**: 상체가 수직일수록 보너스 (`1.0 - |angle|`).

이 접근 방식은 학습 초기부터 높은 성능을 보장하며, 리듬 보상을 통해 점진적으로 자세를 교정하는 효과가 있다.

---

## 4. 알고리즘 및 하이퍼파라미터

### 4.1 알고리즘: PPO (Proximal Policy Optimization)
`Stable-Baselines3` 라이브러리를 사용하였으며, Fine-tuning 시에는 **Learning Rate를 낮춰(1e-4)** 기존 지식의 손실(Catastrophic Forgetting)을 방지하였다.

### 4.2 하이퍼파라미터
*   **Learning Rate**: 1e-4 (Fine-tuning)
*   **Batch Size**: 64
*   **n_steps**: 2048
*   **Total Timesteps**: 1,000,000 (Fine-tuning)
*   **Threshold**: 1800점 (스타일 점수 포함으로 기준 상향)

---

## 5. 실험 결과 (Experimental Results)

### 5.1 학습 곡선 (Training Curve)
*(여기에 `doc/training_curves.png` 이미지가 들어갈 자리입니다)*
전이 학습 덕분에 시작부터 높은 점수(300점 이상)에서 출발하며, 리듬 보상이 추가됨에 따라 점수가 1800점대까지 급상승하여 수렴한다.

### 5.2 시나리오별 성능 평가 (Evaluation) (Mean Reward / 10 Episodes)

| 환경 모드 | V7 (순정/Baseline) | **V9 (최종/Proposed)** | 개선율 |
|---|---|---|---|
| **Normal** | 301.04 | **1803.87*** | (Style 점수 포함) |
| **Heavy** | 286.10 | **1806.96*** | **매우 강건함** |
| **Slippery** | 57.24 | **1748.16*** | **안정성 대폭 향상** |

*(*주: V9의 점수는 리듬/자세 보너스가 포함되어 있어 V7과 직접 비교는 어려우나, 주행 안정성 면에서 압도적임).*

### 5.3 시각적 분석
*   **V7 모델**: 상체를 구부정하게 숙이고 빠르게 미끄러지듯 이동함.
*   **V9 모델**: 상체를 꼿꼿이 세우고(Upright), 일정한 리듬을 유지하며 안정적으로 주행함. 빙판길에서도 자세가 무너지지 않음.

---

## 6. 결론 및 토의

### 6.1 결론
본 프로젝트는 단순한 보행 성공(V7)을 넘어, **인간 친화적인 보행 품질(V9)**을 달성하는 데 성공하였다. 특히 학습된 모델을 재사용하는 **전이 학습(Transfer Learning)** 기법은 학습 시간을 단축시키면서도 새로운 목표(스타일 교정)를 달성하는 데 매우 효과적이었다.

### 6.2 보완점
*   여전히 급격한 기동 시에는 에너지 효율을 위해 한 발을 끄는 현상이 관찰된다. 이를 완벽히 제거하기 위해서는 관절 토크에 대한 더 정밀한 제약 조건이 필요할 것으로 보인다.

---
**[별첨]**
*   소스 코드: `custom_walker.py` (V9 구현)
*   실행 방법: `README.md` 참고
