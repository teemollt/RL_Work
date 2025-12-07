@echo off
chcp 65001
echo ===================================================
echo  [Conda Train] Conda 환경(rl_walker)에서 학습 시작
echo ===================================================

:: Conda 활성화 (Anaconda/Miniconda Path가 설정되어 있다고 가정)
:: 만약 conda 명령어가 없다면 Anaconda Prompt에서 실행하거나 Path 설정 필요
call conda activate rl_walker

if %errorlevel% neq 0 (
    echo [오류] 'rl_walker' 환경을 활성화하지 못했습니다.
    echo.
    echo 1. Conda가 설치되어 있는지 확인하세요.
    echo 2. 'conda env list'로 rl_walker 환경이 존재하는지 확인하세요.
    pause
    exit /b
)

:: Python 실행
echo 활성화된 환경에서 학습 스크립트를 실행합니다...
python run_walker.py --mode normal --timesteps 500000 --seeds 42 --eval

echo.
echo ===================================================
echo  학습 및 평가 완료!
echo ===================================================
pause
