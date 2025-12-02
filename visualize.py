import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

"""
visualize.py
실험 결과를 시각화하여 논문 및 보고서용 그래프를 생성하는 스크립트.
Seaborn을 활용하여 출판 가능한 수준(Publication-quality)의 플롯을 생성함.
"""

def load_results(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        print(f"Error: Result file not found at {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records from {csv_path}")
    return df

def plot_training_curves(df: pd.DataFrame, save_path: str = "training_curves.png"):
    """학습 곡선(Learning Curve) 시각화: 에피소드별 보상 추이"""
    train_df = df[df["phase"] == "training"].copy()
    
    if len(train_df) == 0:
        return
    
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))
    
    # 시드별 학습 곡선 도시
    for seed in train_df["seed"].unique():
        seed_data = train_df[train_df["seed"] == seed]
        
        # 가독성을 위해 이동 평균(Moving Average) 적용
        window = max(1, len(seed_data) // 20)
        smoothed = seed_data["reward"].rolling(window=window, min_periods=1).mean()
        
        plt.plot(seed_data["episode"], smoothed, label=f'Seed {seed}', linewidth=2)
    
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Reward", fontsize=12)
    plt.title("Training Progress: Reward over Episodes", fontsize=14, fontweight='bold')
    plt.legend()
    
    plt.savefig(save_path, dpi=300)
    print(f"Training curve saved: {save_path}")
    plt.close()

def plot_evaluation_comparison(df: pd.DataFrame, save_path: str = "result_graph.png"):
    """
    강건성 평가(Robustness Evaluation) 결과 비교.
    각 환경 모드(Normal, Heavy, Slippery)에 대한 평균 보상과 표준편차를 시각화함.
    """
    eval_df = df[df["phase"] == "evaluation"].copy()
    
    if len(eval_df) == 0:
        return
    
    # 통계치 계산 (Mean & Std Dev)
    stats = eval_df.groupby("mode")["reward"].agg(["mean", "std"]).reset_index()
    stats = stats.sort_values("mean", ascending=False)
    
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    
    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    
    # Bar Plot with Error Bars (Standard Deviation)
    plt.bar(
        stats["mode"], 
        stats["mean"], 
        yerr=stats["std"],
        capsize=10,
        color=colors[:len(stats)],
        alpha=0.8,
        edgecolor='black'
    )
    
    plt.xlabel("Environment Mode", fontsize=12)
    plt.ylabel("Average Reward", fontsize=12)
    plt.title("Robustness Evaluation across Environments", fontsize=14, fontweight='bold')
    
    # 수치 텍스트 추가
    for i, row in stats.iterrows():
        plt.text(i, row['mean'] + row['std'] + 5, 
                 f"{row['mean']:.1f}", 
                 ha='center', fontsize=11, fontweight='bold')
    
    plt.savefig(save_path, dpi=300)
    print(f"Evaluation graph saved: {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="walker_results.csv")
    args = parser.parse_args()
    
    df = load_results(args.csv)
    if df is not None:
        plot_training_curves(df)
        plot_evaluation_comparison(df)

if __name__ == "__main__":
    main()
