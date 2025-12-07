@echo off
chcp 65001
echo ===================================================
echo  [Result Graph] 학습 결과 그래프 그리기
echo ===================================================

:: Conda 활성화
call conda activate rl_walker

if %errorlevel% neq 0 (
    echo [오류] Conda 환경 rl_walker 을 찾을 수 없습니다.
    pause
    exit /b
)

if not exist walker_results.csv (
    echo [오류] walker_results.csv 파일이 없습니다.
    echo 학습을 먼저 진행해주세요 - conda_train.bat 실행
    pause
    exit /b
)

echo.
echo 결과 데이터로 그래프를 생성합니다...
echo.

python visualize.py --csv walker_results.csv

echo.
echo ===================================================
echo  차트 생성 완료!
echo  1. 학습 곡선: training_curves.png
echo  2. 결과 요약: result_graph.png
echo  3. 에너지 효율: energy_graph.png
echo ===================================================
pause
