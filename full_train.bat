@echo off
REM 전체 학습 스크립트 - 완전한 학습 파이프라인

echo ============================================================
echo Walker 학습 - 전체 학습 (100K 타임스텝)
echo 하드웨어에 따라 30-60분 소요
echo ============================================================
echo.

REM 2개 시드로 학습하고 모든 모드에서 평가
python run_walker.py --mode normal --timesteps 100000 --seeds 42 100 --eval

echo.
echo ============================================================
echo 학습 완료! 결과 저장 위치:
echo   - walker_results.csv (학습/평가 데이터)
echo   - models/ (학습된 모델)
echo ============================================================
echo.
echo 다음 단계:
echo   1. 그래프 생성: python visualize.py
echo   2. 영상 녹화: python record_video.py --model models/ppo_walker_normal_seed42
echo.

pause
