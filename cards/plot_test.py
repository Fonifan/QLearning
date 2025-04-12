import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def plot_results(csv_file="card_test_results.csv"):
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' not found!")
        return False

    df = pd.read_csv(csv_file)
    df = df.sort_values(by=["Opponent", "Episode"])

    df['cumulative_mean'] = df.groupby('Opponent')['Win Rate'] \
                              .expanding().mean().reset_index(level=0, drop=True)
    
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(
        x="Episode",
        y="Win Rate",
        hue="Opponent",
        data=df,
        marker='o'
    )
    
    sns.lineplot(
        x="Episode",
        y="cumulative_mean",
        hue="Opponent",
        data=df,
        legend=False,
        linestyle='--'
    )
    
    plt.ylabel('Win Rate')
    plt.xlabel('Episode')
    plt.title('Q-Learning Agent Win Rate Performance')
    
    plt.tight_layout()
    plt.savefig('winrate_progress.png', dpi=300)
    plt.show()

def plot_winrates(csv_file="card_winrate.csv"):
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' not found!")
        return False

    df = pd.read_csv(csv_file)
    df = df.sort_values(by="Episode")
    df['cumulative_mean'] = df['Win Rate'].expanding().mean().reset_index(drop=True)

    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(
        x="Episode",
        y="Win Rate",
        data=df,
        marker='o',
        label="Win Rate"
    )
    
    sns.lineplot(
        x="Episode",
        y="cumulative_mean",
        data=df,
        linestyle='--',
        label="Cumulative Mean"
    )
    
    plt.ylabel('Win Rate')
    plt.xlabel('Episode')
    plt.title('Q-Learning Agent Win Rates in Training')
    
    plt.tight_layout()
    plt.savefig('winrate_training.png', dpi=300)
    plt.show()
    
if __name__ == "__main__":
    plot_results()
    plot_winrates()