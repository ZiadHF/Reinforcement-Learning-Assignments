"""
Visualization Script for DQN/DDQN Assignment Report
Run this after training to generate publication-quality plots
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

def load_results(results_file='results.json'):
    """Load training results from JSON file"""
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            return json.load(f)
    return None

def plot_training_comparison(dqn_rewards, ddqn_rewards, env_name, save_dir='plots'):
    """Compare DQN vs DDQN training curves"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot with smoothing
    window = 50
    dqn_smooth = np.convolve(dqn_rewards, np.ones(window)/window, mode='valid')
    ddqn_smooth = np.convolve(ddqn_rewards, np.ones(window)/window, mode='valid')
    
    ax.plot(dqn_rewards, alpha=0.3, color=colors[0], label='DQN (raw)')
    ax.plot(dqn_smooth, color=colors[0], linewidth=2, label='DQN (smoothed)')
    ax.plot(ddqn_rewards, alpha=0.3, color=colors[1], label='DDQN (raw)')
    ax.plot(ddqn_smooth, color=colors[1], linewidth=2, label='DDQN (smoothed)')
    
    ax.set_xlabel('Episode', fontsize=14)
    ax.set_ylabel('Reward', fontsize=14)
    ax.set_title(f'Training Curves: {env_name}', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{env_name}_training_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_evaluation_stability(eval_rewards, env_name, algo_name, save_dir='plots'):
    """Plot evaluation episode rewards to show stability"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Rewards over episodes
    ax1.plot(eval_rewards, marker='o', markersize=3, linewidth=1, color=colors[0])
    ax1.axhline(np.mean(eval_rewards), color='r', linestyle='--', label=f'Mean: {np.mean(eval_rewards):.2f}')
    ax1.fill_between(range(len(eval_rewards)), 
                     np.mean(eval_rewards) - np.std(eval_rewards),
                     np.mean(eval_rewards) + np.std(eval_rewards),
                     alpha=0.2, color='r', label=f'±1 Std: {np.std(eval_rewards):.2f}')
    ax1.set_xlabel('Test Episode', fontsize=12)
    ax1.set_ylabel('Reward', fontsize=12)
    ax1.set_title(f'{algo_name} Stability: {env_name}', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Histogram
    ax2.hist(eval_rewards, bins=20, color=colors[1], alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(eval_rewards), color='r', linestyle='--', linewidth=2, label='Mean')
    ax2.set_xlabel('Reward', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Reward Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{env_name}_{algo_name}_stability.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_hyperparameter_comparison(results_dict, param_name, env_name, save_dir='plots'):
    """
    Plot effect of different hyperparameter values
    
    results_dict: {param_value: [list of rewards], ...}
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    param_values = []
    mean_rewards = []
    std_rewards = []
    
    for value, rewards in sorted(results_dict.items()):
        param_values.append(str(value))
        mean_rewards.append(np.mean(rewards))
        std_rewards.append(np.std(rewards))
    
    x_pos = np.arange(len(param_values))
    ax.bar(x_pos, mean_rewards, yerr=std_rewards, capsize=5, 
           color=colors[2], alpha=0.7, edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(param_values)
    ax.set_xlabel(param_name, fontsize=14)
    ax.set_ylabel('Average Reward', fontsize=14)
    ax.set_title(f'Effect of {param_name} on {env_name}', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{env_name}_{param_name}_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_all_environments_comparison(results, save_dir='plots'):
    """Create summary plot comparing all environments"""
    os.makedirs(save_dir, exist_ok=True)
    
    environments = list(results.keys())
    dqn_means = [results[env]['DQN']['mean'] for env in environments]
    dqn_stds = [results[env]['DQN']['std'] for env in environments]
    ddqn_means = [results[env]['DDQN']['mean'] for env in environments]
    ddqn_stds = [results[env]['DDQN']['std'] for env in environments]
    
    x = np.arange(len(environments))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars1 = ax.bar(x - width/2, dqn_means, width, yerr=dqn_stds, 
                   label='DQN', color=colors[0], alpha=0.8, capsize=5, edgecolor='black')
    bars2 = ax.bar(x + width/2, ddqn_means, width, yerr=ddqn_stds,
                   label='DDQN', color=colors[1], alpha=0.8, capsize=5, edgecolor='black')
    
    ax.set_xlabel('Environment', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Reward (100 episodes)', fontsize=14, fontweight='bold')
    ax.set_title('DQN vs DDQN Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(environments, rotation=15, ha='right')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
    
    autolabel(bars1)
    autolabel(bars2)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/all_environments_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_loss_curves(losses, env_name, algo_name, save_dir='plots'):
    """Plot training loss over episodes"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Apply smoothing
    window = 50
    if len(losses) > window:
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        ax.plot(losses, alpha=0.3, color=colors[3], label='Loss (raw)')
        ax.plot(range(window-1, len(losses)), smoothed, color=colors[3], 
                linewidth=2, label='Loss (smoothed)')
    else:
        ax.plot(losses, color=colors[3], linewidth=2, label='Loss')
    
    ax.set_xlabel('Episode', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.set_title(f'{algo_name} Training Loss: {env_name}', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale often better for loss
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{env_name}_{algo_name}_loss.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_table(results, save_path='summary_table.txt'):
    """Create a LaTeX-formatted summary table"""
    
    with open(save_path, 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{|l|c|c|c|}\n")
        f.write("\\hline\n")
        f.write("\\textbf{Environment} & \\textbf{DQN} & \\textbf{DDQN} & \\textbf{Improvement} \\\\\n")
        f.write("\\hline\n")
        
        for env_name, data in results.items():
            dqn_mean = data['DQN']['mean']
            dqn_std = data['DQN']['std']
            ddqn_mean = data['DDQN']['mean']
            ddqn_std = data['DDQN']['std']
            improvement = ((ddqn_mean - dqn_mean) / abs(dqn_mean)) * 100
            
            f.write(f"{env_name} & ${dqn_mean:.1f} \\pm {dqn_std:.1f}$ & "
                   f"${ddqn_mean:.1f} \\pm {ddqn_std:.1f}$ & "
                   f"{improvement:+.1f}\\% \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Performance Comparison: DQN vs DDQN (Mean $\\pm$ Std over 100 episodes)}\n")
        f.write("\\label{tab:results}\n")
        f.write("\\end{table}\n")
    
    print(f"LaTeX table saved to {save_path}")

# Example usage (you'll need to adapt this to your actual data)
if __name__ == "__main__":
    print("Visualization Script for DQN/DDQN Assignment")
    print("=" * 60)
    print()
    print("This script provides functions to create publication-quality plots.")
    print()
    print("Example Usage:")
    print("-" * 60)
    print("""
    # After training, save your results:
    results = {
        'CartPole-v1': {
            'DQN': {'mean': 485.3, 'std': 12.5, 'rewards': [...]},
            'DDQN': {'mean': 495.1, 'std': 8.2, 'rewards': [...]}
        },
        # ... other environments
    }
    
    # Create plots:
    plot_training_comparison(dqn_rewards, ddqn_rewards, 'CartPole-v1')
    plot_evaluation_stability(eval_rewards, 'CartPole-v1', 'DQN')
    plot_all_environments_comparison(results)
    create_summary_table(results)
    
    # For hyperparameter analysis:
    gamma_results = {
        0.95: [450, 460, 455, ...],  # 100 episode rewards
        0.99: [485, 490, 488, ...],
        0.999: [470, 475, 472, ...]
    }
    plot_hyperparameter_comparison(gamma_results, 'GAMMA', 'CartPole-v1')
    """)
    print("-" * 60)
    print()
    print("All plots will be saved in the 'plots/' directory.")
    print("Customize colors, styles, and layouts as needed for your report!")
