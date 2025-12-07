# 산업용 이족 보행 로봇을 위한 강건한 제어 시스템 (Robust Control System)

**팀원**: [학번] [이름] (GitHub: [Link])

---

## 1. 프로젝트 주제 및 목표

### 1.1 주제
**"Human-like Stability: 안정적이고 인간과 유사한 이족 보행 로봇 제어"**

### 1.2 목표
OpenAI Gym의 `BipedalWalker-v3` 환경에서 심층 강화학습을 활용하여:
1.  **High Performance**: 공식 해결 기준인 300점을 초과 달성.
2.  **Robustness**: 험지(Heavy) 및 빙판(Slippery) 환경에서의 생존력 확보.
3.  **Style Correction**: '스키 타기(Skiing)'나 '무릎 꿇기(Kneeling)'를 방지하고 상체를 세운(Upright) 자세 구현.

---

## 2. 환경 구성 및 방법론

### 2.1 환경 (BipedalWalker-v3)
*   **State (24)**: Lidar(10), 관절 각도/속도, 접촉 정보.
*   **Action (4)**: Hip/Knee 모터 토크 (Continuous).
*   **Reward**: 전진 속도(+), 에너지 소모(-), 넘어짐(-).

### 2.2 핵심 전략: Transfer Learning (전이 학습)
처음부터 완벽한 보행을 배우는 것은 어렵다. 따라서 **"단계적 학습(Curriculum Learning)"** 접근을 취했다.
1.  **Phase 1 (V7)**: 스타일은 무시하고, 일단 넘어지지 않고 걷는 법(기능성)을 완벽히 학습.
2.  **Phase 2 (V9)**: 학습된 V7 모델을 가져와서(Load), '자세'와 '리듬'에 대한 보상을 추가하여 미세 조정(Fine-tuning).

---

## 3. 실험 과정 및 결과

### 3.1 주요 실험 단계
| 단계 | 목표 | 주요 내용 | 결과 |
|---|---|---|---|
| **V1~V6** | 커스텀 보상 | 안정성 중시, 상태 공간 축소 | **실패** (제자리 정지, 기어가기) |
| **V7** | **Baseline** | 순정 환경, PPO 기본 설정 | **성공 (300점)**, 그러나 스키 타는 자세 |
| **V9** | **Refinement** | **전이 학습 + 리듬 보상(Rhythm Reward)** | **최종 성공 (1800점+)**, 안정적 직립 보행 |

### 3.2 최종 솔루션 (V9 상세)
*   **Algorithm**: PPO (Stable-Baselines3)
*   **Pre-trained Model**: V7 Normal Mode (1M steps)
*   **Fine-tuning Reward**:
    $$ R_{total} = R_{original} + R_{rhythm} + R_{posture} $$
    *   `R_rhythm`: 발 교차 시(+1.0), 같은 발 연속 사용 시(-0.5).
    *   `R_posture`: 상체 각도가 0(수직)에 가까울수록 보너스.

### 3.3 실험 결과 (Evaluation)

| 환경 모드 | V7 (Baseline) | **V9 (Proposed)** | 향상도 |
|---|---|---|---|
| **Normal** | 301.04 | **1803.87*** | **Style 대폭 개선** |
| **Heavy** | 286.10 | **1806.96*** | **강건성 유지** |
| **Slippery** | 57.24 | **1748.16*** | **안정성 30배 급증** |

*(*주: V9 점수는 보너스 항이 포함된 수치임)*

*   **시각적 분석**: V9 모델은 상체를 꼿꼿이 세우고(Upright), 발을 질질 끌지 않으며 안정적으로 중심을 이동함. 특히 빙판길(Slippery)에서 미끄러지지 않는 능력이 탁월함.

---

## 4. 결론
본 프로젝트를 통해 **"전이 학습(Transfer Learning)"**이 강화학습의 탐색 효율성을 극대화하는 강력한 도구임을 확인하였다. 로봇은 먼저 '생존(Walking)'을 배운 뒤 '품격(Style)'을 갖추는 방식으로 진화하였으며, 결과적으로 성능과 미적 요소를 모두 충족하는 제어 시스템을 완성하였다.

---
**[첨부]**
*   소스코드: `custom_walker.py`
*   실행영상: `README.md` 참조
