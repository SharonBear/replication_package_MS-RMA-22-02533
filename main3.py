# main3.py
import numpy as np
import matplotlib.pyplot as plt
from restartq_ucb_multiagent import restartq_ucb_multiagent

M = 5000
H = 5
test_num = 5

reward_runs = []
regret_runs = []

for _ in range(test_num):
    rewards, regrets = restartq_ucb_multiagent(M, H)
    reward_runs.append(np.cumsum(rewards))
    regret_runs.append(np.cumsum(regrets))

reward_runs = np.array(reward_runs)
regret_runs = np.array(regret_runs)

# 平均與誤差計算
def compute_stats(data):
    mean = np.mean(data, axis=0)
    err = 1.96 * np.std(data, axis=0) / np.sqrt(data.shape[0])
    return mean, err

mean_reward, err_reward = compute_stats(reward_runs)
mean_regret, err_regret = compute_stats(regret_runs)

x = np.arange(M)

# 畫 cumulative reward
plt.figure()
plt.plot(x, mean_reward, label="Cumulative Reward")
plt.fill_between(x, mean_reward - err_reward, mean_reward + err_reward, alpha=0.2)
plt.xlabel("Episodes")
plt.ylabel("Cumulative Reward")
plt.title("Multi-Agent RestartQ-UCB - Reward")
plt.legend()
plt.grid(True)

# 畫 cumulative regret
plt.figure()
plt.plot(x, mean_regret, label="Cumulative Regret", color='orange')
plt.fill_between(x, mean_regret - err_regret, mean_regret + err_regret, color='orange', alpha=0.2)
plt.xlabel("Episodes")
plt.ylabel("Cumulative Regret")
plt.title("Multi-Agent RestartQ-UCB - Regret")
plt.legend()
plt.grid(True)

plt.show()
