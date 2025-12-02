@echo off
REM 빠른 학습 스크립트 - 짧은 학습 테스트 실행

echo ============================================================
echo Walker 학습 - 빠른 테스트 (1000 타임스텝)
echo ============================================================
echo.

REM 단일 시드로 빠른 학습 실행
python run_walker.py --mode normal --timesteps 1000 --seeds 42

echo.
echo ============================================================
echo 빠른 테스트 완료! walker_results.csv를 확인하세요
echo ============================================================
echo.
echo 전체 학습(100K 타임스텝)을 실행하려면:
echo   python run_walker.py --mode normal --timesteps 100000 --seeds 42 100 --eval
echo.

pause
