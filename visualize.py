import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import numpy as np

"""
visualize.py
실험 결과를 시각화하여 보고서용 고품질 그래프를 생성하는 스크립트.
Seaborn을 활용하여 Academic Style의 그래프를 생성함.
"""

# 학술적인 느낌의 스타일 설정
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
# 한글 폰트 설정이 필요할 수 있으나, 영문으로 작성하되 깔끔하게.

def load_results(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        print(f"Error: Result file not found at {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records from {csv_path}")
    return df

def plot_training_curves(df: pd.DataFrame, save_path: str = "training_curves.png"):
    """
    학습 곡선(Learning Curve) 시각화
    - 시드별 평균 및 표준편차(Confidence Interval) 표현
    - 이동 평균(Rolling Mean)을 사용하여 트렌드 강조
    """
    train_df = df[df["phase"] == "training"].copy()
    
    if len(train_df) == 0:
        return
    
    plt.figure(figsize=(10, 6))
    
    # 데이터 스무딩
    # 윈도우 크기는 전체 데이터 길이에 비례하되 최소 10
    window_size = max(10, int(len(train_df) / 100))
    train_df["smoothed_reward"] = train_df.groupby("seed")["reward"].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).mean()
    )
    
    # Line Plot with CI (95% 신뢰구간)
    # x="episode" 대신 인덱스를 다시 맞춰서 사용할 수도 있음
    sns.lineplot(
        data=train_df, 
        x="episode", 
        y="smoothed_reward",
        hue="mode",
        errorbar="sd",  # 표준편차로 음영 처리
        palette="viridis",
        linewidth=2
    )
    
    # 목표 점수 (Solved 기준)
    plt.axhline(y=300, color='r', linestyle='--', label='Solved (300)', alpha=0.7)
    
    plt.xlabel("Episodes", fontweight='bold')
    plt.ylabel("Reward (Smoothed)", fontweight='bold')
    plt.title("Training Progress: Reward over Episodes", fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curve saved: {save_path}")
    plt.close()

def plot_evaluation_comparison(df: pd.DataFrame, save_path: str = "result_graph.png"):
    """
    평가 결과 비교 (Robustness Check)
    - Bar Plot: 평균 및 에러바
    - Strip Plot: 개별 시드의 분포
    """
    eval_df = df[df["phase"] == "evaluation"].copy()
    
    if len(eval_df) == 0:
        return
    
    plt.figure(figsize=(10, 6))
    
    # 색상 팔레트
    palette = sns.color_palette("muted")
    
    # 1. Bar Plot (평균)
    ax = sns.barplot(
        data=eval_df,
        x="mode",
        y="reward",
        errorbar="sd",
        capsize=0.1,
        palette=palette,
        edgecolor="black",
        alpha=0.8
    )
    
    # 2. Strip Plot (개별 데이터 포인트)
    sns.stripplot(
        data=eval_df,
        x="mode",
        y="reward",
        color="black",
        alpha=0.4,
        jitter=0.1,
        dodge=True
    )
    
    # 수치 텍스트 추가
    means = eval_df.groupby("mode")["reward"].mean()
    for i, mode in enumerate(eval_df["mode"].unique()):
        # seaborn은 정렬 순서가 다를 수 있으므로 주의. 여기선 간단히 처리
        if mode in means:
            val = means[mode]
            ax.text(i, val + 10, f"{val:.1f}", ha='center', fontweight='bold', color='black')

    plt.xlabel("Environment Mode", fontweight='bold')
    plt.ylabel("Average Reward", fontweight='bold')
    plt.title("Performance Evaluation (Robustness Check)", fontsize=14, fontweight='bold', pad=15)
    plt.axhline(y=300, color='r', linestyle='--', alpha=0.5, label='Solved Threshold')
    
    plt.ylim(bottom=-100) # 너무 낮은 점수는 자르거나 조정
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Evaluation graph saved: {save_path}")
    plt.close()

def plot_energy_efficiency(df: pd.DataFrame, save_path: str = "energy_graph.png"):
    """
    에너지 효율성 분석
    - Reward 대비 Energy 사용량 분석
    """
    eval_df = df[df["phase"] == "evaluation"].copy()
    
    if len(eval_df) == 0:
        return
        
    plt.figure(figsize=(10, 6))
    
    # Box Plot for Energy Distribution
    sns.boxplot(
        data=eval_df,
        x="mode",
        y="energy",
        palette="pastel"
    )
    
    plt.xlabel("Environment Mode", fontweight='bold')
    plt.ylabel("Energy Used (Lower is Better)", fontweight='bold')
    plt.title("Energy Efficiency Analysis", fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Energy graph saved: {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="walker_results.csv")
    args = parser.parse_args()
    
    df = load_results(args.csv)
    if df is not None:
        plot_training_curves(df)
        plot_evaluation_comparison(df)
        plot_energy_efficiency(df)
        print("\nAll visualizations created successfully.")

if __name__ == "__main__":
    main()
