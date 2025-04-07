import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_results(csv_file="card_test_results.csv"):
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' not found!")
        return False
    
    df = pd.read_csv(csv_file)
    
    sns.set_theme(style="darkgrid")
    
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(x="Episode", y="Win Rate", data=df, marker='o', color='blue')
    plt.ylabel('Win Rate')
    plt.xlabel('Episode')
    plt.title('Q-Learning Agent Win Rate Performance')
    
    plt.tight_layout()
    plt.savefig('winrate_progress.png', dpi=300)
    plt.show()
    
    print("Plot saved as 'winrate_progress.png'")
    return True

if __name__ == "__main__":
    plot_results()