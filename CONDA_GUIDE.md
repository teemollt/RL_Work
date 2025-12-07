# Conda 기반 RL 환경(rl_walker) 가이드

이 문서는 `venv` 대신 **Anaconda/Miniconda** 기반의 `rl_walker` 가상환경을 사용하여 강화학습을 수행하기 위한 가이드입니다.

## 1. 환경 개요

현재 설정된 `rl_walker` 환경은 BipedalWalker-v3(Box2D 기반) 강화학습을 위해 다음의 핵심 라이브러리들로 구성됩니다.

### 주요 라이브러리 설명

| 라이브러리 | 용도 | 설명 |
|---|---|---|
| **Gymnasium** | RL 환경 | OpenAI Gym의 유지보수 버전으로, 강화학습 에이전트가 상호작용할 표준 인터페이스 제공 (Box2D 물리 엔진 포함) |
| **Stable-Baselines3** | RL 알고리즘 | PPO, SAC 등 검증된 강화학습 알고리즘 구현체. PyTorch 기반. |
| **Shimmy** | 호환성 | 다양한 환경(Gym 등)을 Gymnasium API로 변환해주는 호환성 레이어 |
| **Swig** | 빌드 도구 | Box2D 물리 엔진을 Python에서 사용하기 위한 C/C++ 인터페이스 컴파일 도구 (Windows 필수) |
| **Pygame** | 렌더링 | 학습 과정 시각화 및 렌더링 지원 |
| **Numpy/Pandas** | 데이터 처리 | 수치 연산 및 학습 로그/결과 데이터 분석 |

---

## 2. 사용 방법

### 2.1. Conda 환경 활성화

터미널(Anaconda Prompt 또는 PowerShell)에서 다음 명령어로 환경을 활성화합니다.

```bash
conda activate rl_walker
```

활성화에 성공하면 프롬프트 앞부분이 `(rl_walker)`로 변경됩니다.

### 2.2. 의존성 업데이트/확인

새로운 라이브러리가 필요하거나 환경을 동기화해야 할 때는 `environment.yml` 파일을 사용합니다.

```bash
# 환경 활성화 상태에서 실행
conda env update --file environment.yml --prune
```

> **참고**: `swig`의 경우 Windows에서 `pip`로 설치 시 문제가 발생할 수 있습니다. 만약 에러가 발생한다면 Conda 명령어로 설치하세요:
> ```bash
> conda install -c conda-forge swig
> ```

---

## 3. 학습 실행

### 3.1. 자동화 스크립트 사용 (`conda_train.bat`)

새로 추가된 `conda_train.bat` 파일을 더블 클릭하거나 터미널에서 실행하면 자동으로 환경을 활성화하고 전체 학습을 진행합니다.

```cmd
.\conda_train.bat
```

### 3.2. 수동 실행

직접 명령어를 입력하여 실행 모드를 세밀하게 조정할 수 있습니다.

```bash
# 기본 학습 (Normal 모드, 10만 스텝, 시드 42/100)
python run_walker.py --mode normal --timesteps 100000 --seeds 42 100 --eval

# 빠른 테스트 (1000 스텝, 시드 42만)
python run_walker.py --mode normal --timesteps 1000 --seeds 42
```

---

## 4. 파일 구조 및 설명

- **`run_walker.py`**: 강화학습 메인 스크립트. PPO 알고리즘을 설정하고 학습 및 저장을 수행합니다.
- **`custom_walker.py`**: 기본 BipedalWalker 환경을 상속받아 커스텀(지형 변경 등) 환경을 정의한 파일입니다.
- **`conda_train.bat`**: Conda 환경 전용 실행 스크립트입니다.

## 5. 자주 발생하는 문제 (Troubleshooting)

**Q. `ImportError: No module named 'Box2D'` 오류가 발생해요.**
A. Gymnasium의 Box2D 의존성이 제대로 설치되지 않은 경우입니다.
`swig`가 설치되어 있는지 확인 후 다음을 재설치해보세요.
```bash
conda install -c conda-forge swig
pip install gymnasium[box2d]
```

**Q. `conda activate` 명령어가 인식이 안 돼요.**
A. 일반 CMD나 PowerShell에서는 Conda 경로가 PATH에 잡혀있지 않을 수 있습니다. **Anaconda Prompt**를 사용하거나, 터미널 실행 시 Conda 초기화를 수행해야 합니다.
