# 산업용 이족 보행 로봇을 위한 강건한 제어 시스템 (Robust Bipedal Walker Control System)

이 저장소는 심층 강화학습(PPO)을 활용하여 `BipedalWalker-v3` 환경에서 안정적이고 인간과 유사한 이족 보행을 구현한 프로젝트입니다.

## 🚀 시작하기 (실행 가이드)

Box2D 물리 엔진의 의존성 문제로 인해, 본 프로젝트는 **Anaconda(또는 Miniconda)** 환경에서의 실행을 강력히 권장합니다.

### 1. 필수 요구사항 (Prerequisites)
- **Anaconda** 또는 **Miniconda** 설치 필요
- **Git** 설치 필요

### 2. 가상환경 설정 (Environment Setup)

1.  **저장소 복제 (Clone)**:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Conda 환경 생성**:
    동봉된 `environment.yml` 파일을 사용하여 필요한 모든 라이브러리를 자동으로 설치합니다.
    ```bash
    conda env create -f environment.yml
    ```

3.  **환경 활성화**:
    ```bash
    conda activate rl_walker
    ```

### 3. 학습 및 실행 (Execution)

Windows 사용자를 위해 간편한 배치 파일(.bat)을 제공합니다.

#### **A. 처음부터 학습 시작 (Train from Scratch)**
최종 설계된 모델(V12 - Swing Phase Enforcement)로 학습을 시작하려면 아래 명령어를 실행하세요.
```bash
conda_train.bat
```
- 기본 설정으로 500,000 스텝 동안 학습을 진행합니다.
- 학습 로그는 `logs/` 폴더에, 모델 체크포인트는 `models/` 폴더에 저장됩니다.

#### **B. 결과 영상 생성 (Visualize)**
학습된 모델이 실제 어떻게 걷는지 영상(GIF)으로 확인하려면:
```bash
conda_record.bat
```
- `videos/` 폴더에 결과 영상이 저장됩니다.
- **Normal** (일반), **Heavy** (무거운 다리), **Slippery** (미끄러운 바닥) 세 가지 시나리오를 자동으로 평가합니다.

#### **C. 학습 결과 그래프 확인 (Plotting)**
보상(Reward) 변화와 에너지 효율성 등을 그래프로 확인하려면:
```bash
conda_result.bat
```
- `training_curves.png`, `energy_graph.png` 등의 이미지 파일이 생성됩니다.

---

## 📂 프로젝트 구조

- `custom_walker.py`: 이족 보행 환경을 수정한 Wrapper 클래스 (V9 로직).
- `run_walker.py`: 학습 메인 스크립트 (Stable-Baselines3 PPO 사용).
- `record_video.py`: 평가 및 영상 생성 스크립트.
- `environment.yml`: Conda 환경 설정 파일.
- `archive_*`: 이전 실험(V1 ~ V8)의 코드와 결과물이 보존된 아카이브 폴더.
- `doc/`: 프로젝트 보고서 및 발표 자료.

---

## 📊 실험 결과 요약 (Results)

### 1. 학습 곡선 (Training Metrics)
전이 학습(Transfer Learning)을 통해 초기부터 높은 성능을 유지하며 스타일을 교정합니다.
![Training Curves](training_curves.png)

### 2. 성능 평가 (Evaluation)
다양한 환경(Normal, Heavy, Slippery)에서의 보상 분포입니다.
![Evaluation Graph](result_graph.png)

### 3. 시연 영상 (Demo)
*(영상이 보이지 않는다면 `conda_record.bat`을 실행하여 생성하세요)*

| Normal Mode | Heavy Mode | Slippery Mode |
|:---:|:---:|:---:|
| ![Normal](videos/walker_normal.gif) | ![Heavy](videos/walker_heavy.gif) | ![Slippery](videos/walker_slippery.gif) |

