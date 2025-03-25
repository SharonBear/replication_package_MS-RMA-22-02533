import numpy as np
from matplotlib import pyplot as plt
import argparse

# 匯入你的演算法
from RestartQ_UCB import restartq_ucb
from LSVI_UCB_Restart import lsvi_ucb_restart
from Q_Learning_UCB import restartq_ucb_no_restart
from Epsilon_Greedy import q_learning_epsilon_greedy
from Double_RestartQ_UCB import double_restartq_ucb


def smooth(y, sm=1):
    if sm > 1:
        ker = np.ones(sm) * 1.0 / sm
        y = np.convolve(ker, y, "same")
    return y

# 環境與實驗參數
A = 2
H = 5
S = 2 * H
M = 5000
variation = 50

# 設定演算法
func_list = [restartq_ucb, lsvi_ucb_restart, restartq_ucb_no_restart, q_learning_epsilon_greedy, double_restartq_ucb]
algo_name_list = ['RestartQ-UCB', 'LSVI-UCB-Restart', 'Q-Learning UCB', 'Epsilon-Greedy', 'Double-Restart Q-UCB']
colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c', '#9467bd']

parser = argparse.ArgumentParser()
parser.add_argument('--metric', type=str, default='both', choices=['reward', 'regret', 'both'])
args = parser.parse_args()

x = range(M)
test_num = 5

# 初始化儲存所有演算法的 reward/regret
reward_means = []
reward_errs = []
regret_means = []
regret_errs = []

for algo_id, algo_func in enumerate(func_list):
    rewards_all = []
    regrets_all = []
    for test_id in range(test_num):
        reward, regret = algo_func(S, A, M, H, variation)
        rewards_all.append(np.cumsum(reward))
        regrets_all.append(np.cumsum(regret))

    rewards_all = np.array(rewards_all)
    regrets_all = np.array(regrets_all)

    def compute_mean_and_err(data):
        mean = np.mean(data, axis=0)
        err = 1.96 * np.std(data, axis=0) / np.sqrt(test_num)
        return smooth(mean, 3), smooth(err, 3)

    mean_r, err_r = compute_mean_and_err(rewards_all)
    mean_g, err_g = compute_mean_and_err(regrets_all)

    reward_means.append(mean_r)
    reward_errs.append(err_r)
    regret_means.append(mean_g)
    regret_errs.append(err_g)

# 畫圖函數
def plot_all(metric_name, means, errs):
    fig, ax = plt.subplots()
    for i, (mean, err) in enumerate(zip(means, errs)):
        ax.plot(x[3:-3], mean[3:-3], label=algo_name_list[i], color=colors[i])
        ax.fill_between(x[3:-3], mean[3:-3] - err[3:-3], mean[3:-3] + err[3:-3], color=colors[i], alpha=0.1)
    ax.set_xlabel('Episodes', fontsize='x-large')
    ax.set_ylabel(f'Cumulative {metric_name.capitalize()}', fontsize='x-large')
    ax.legend(loc='upper left', fontsize='large')
    plt.title(f'Cumulative {metric_name.capitalize()} Comparison')
    plt.tight_layout()
    plt.show()

if args.metric in ['reward', 'both']:
    plot_all('reward', reward_means, reward_errs)
if args.metric in ['regret', 'both']:
    plot_all('regret', regret_means, regret_errs)
