@echo off
chcp 65001
echo ===================================================
echo  [Record Video] 학습된 모델로 영상(GIF) 만들기
echo ===================================================

:: Conda 활성화
call conda activate rl_walker

if %errorlevel% neq 0 (
    echo [오류] Conda 환경 rl_walker 를 찾을 수 없습니다.
    pause
    exit /b
)

:: 1. 성장 과정 영상 (Progression Video) 생성
echo.
echo [1/2] 성장 과정 영상 생성 (Training Progression)
echo ---------------------------------------------------

:: CSV 파일에서 가장 높은 점수를 기록한 시드(Seed) 찾기
python -c "import pandas as pd; df = pd.read_csv('walker_results.csv'); best_row = df.loc[df['reward'].idxmax()]; print(int(best_row['seed']))" > best_seed.txt
set /p BEST_SEED=<best_seed.txt
del best_seed.txt

echo [분석] 가장 우수한 성능을 보인 Seed: %BEST_SEED%
set CHECKPOINT_DIR=models\checkpoints_%BEST_SEED%

if exist %CHECKPOINT_DIR% (
    echo 체크포인트 경로: %CHECKPOINT_DIR%
    python record_video.py --checkpoint_dir %CHECKPOINT_DIR% --output videos
) else (
    echo [주의] 체크포인트 폴더를 찾을 수 없습니다: %CHECKPOINT_DIR%
    echo 학습 과정 영상은 생성되지 않습니다.
)

:: 2. 평가 영상 (Evaluation Video) 생성
echo.
echo [2/2] 평가 영상 생성 (Normal, Heavy, Slippery)
echo ---------------------------------------------------

:: 우선 베스트 모델을 찾습니다.
if exist models\best_model.zip (
    set MODEL_PATH=models\best_model.zip
    echo [알림] 최고 성능 모델 best_model.zip 을 발견하여 사용합니다.
) else (
    :: 없다면 Seed 42 모델 사용
    set MODEL_PATH=models\ppo_walker_normal_seed42.zip
    echo [알림] 베스트 모델이 없어 Seed 42 모델을 사용합니다.
)

if exist %MODEL_PATH% (
    echo 모델 경로: %MODEL_PATH%
    python record_video.py --model %MODEL_PATH% --output videos
) else (
    echo [오류] 모델 파일을 찾을 수 없습니다: %MODEL_PATH%
    echo 학습을 먼저 진행해주세요 - conda_train.bat 실행
)

echo.
echo ===================================================
echo  모든 작업 완료! 결과물은 videos 폴더를 확인하세요.
echo ===================================================
pause
