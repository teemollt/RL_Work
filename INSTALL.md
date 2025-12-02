# Installation Guide

본 프로젝트의 실험 환경 구성을 위한 가이드입니다.
Windows 환경에서의 `box2d-py` 빌드 호환성 문제를 해결하기 위한 방법을 포함
## Prerequisites
- Python 3.8+
- Anaconda or Miniconda (Recommended for Windows users)

## Installation Steps

### 1. Create Virtual Environment
실험의 재현성을 위해 독립된 가상환경 사용.
```bash
conda create -n rl_walker python=3.9
conda activate rl_walker
```

### 2. Install Dependencies
기본 의존성 패키지를 설치
```bash
pip install gymnasium stable-baselines3 shimmy pandas matplotlib seaborn imageio
```

### 3. Install Box2D (Physics Engine)
Windows 환경에서는 pip를 통한 소스 빌드가 실패할 수 있으므로, `conda-forge` 채널을 이용하거나 미리 빌드된 wheel 파일을 사용하는 것을 권장

**Method A: Conda (Recommended)**
```bash
conda install -c conda-forge box2d-py
```

**Method B: Pip**
```bash
pip install gymnasium[box2d]
```

## Verification
설치가 완료되면 아래 명령어로 환경 구성을 검증
```bash
python custom_walker.py
```
오류 없이 Observation Space와 Action Space 정보가 출력되면 정상
