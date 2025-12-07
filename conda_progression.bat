@echo off
chcp 65001
echo ===================================================
echo  [Progression Video] AI 성장 과정 영상 만들기
echo ===================================================

:: Conda 활성화
call conda activate rl_walker

if %errorlevel% neq 0 (
    echo [오류] Conda 환경 rl_walker 을 찾을 수 없습니다.
    pause
    exit /b
)

:: 체크포인트 폴더 확인 (Seed 42 기준)
set CHECKPOINT_DIR=models\checkpoints_42

if not exist %CHECKPOINT_DIR% (
    echo [오류] 체크포인트 폴더를 찾을 수 없습니다: %CHECKPOINT_DIR%
    echo 학습 시 체크포인트가 저장되지 않았거나, 아직 학습을 진행하지 않았습니다.
    pause
    exit /b
)

echo.
echo 체크포인트 경로: %CHECKPOINT_DIR%
echo 성장 과정을 영상으로 합칩니다...
echo.

python record_video.py --checkpoint_dir %CHECKPOINT_DIR% --output videos

echo.
echo ===================================================
echo  작업 완료! 
echo  생성된 영상: videos\training_progression.gif
echo ===================================================
pause
