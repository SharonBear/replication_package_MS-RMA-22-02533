import numpy as np
from matplotlib import pyplot as plt
import os
import math
from RestartQ_UCB import restartq_ucb
from LSVI_UCB_Restart import lsvi_ucb_restart
from Q_Learning_UCB import restartq_ucb_no_restart
from Epsilon_Greedy import q_learning_epsilon_greedy
from Double_RestartQ_UCB import double_restartq_ucb
import time 
import argparse


def smooth(y, sm=1):
    if sm > 1:
        ker = np.ones(sm) * 1.0 / sm
        y = np.convolve(ker, y, "same")
    return y


A = 2
H = 5
S = 2 * H
M = 5000
prob_threshold = 0.98
variation = 50
abrupt = False
if abrupt:
    prefix = 'abrupt-'
else:
    prefix = 'gradual-'

func_list = [restartq_ucb, lsvi_ucb_restart, restartq_ucb_no_restart, q_learning_epsilon_greedy, double_restartq_ucb]
algo_name_list = ['RestartQ-UCB', 'LSVI-UCB-Restart', 'Q-Learning UCB', 'Epsilon-Greedy', 'Double-Restart Q-UCB']

parser = argparse.ArgumentParser()
parser.add_argument('--alg', type=int, default=2)
args = parser.parse_args()
x = range(M)
fig, ax = plt.subplots()
algo_id = args.alg
test_num = 5

a = []
for test_id in range(test_num):
    ans = np.cumsum(func_list[algo_id](S, A, M, H, variation))
    a.append(ans)

a = np.array(a)
err = []
mean = []

for j in range(M):
    tmp  = a[:, j]
    err.append(1.96 * np.std(tmp) / np.sqrt(len(tmp)))
    mean.append(np.mean(tmp))

err = np.array(err)
mean = np.array(mean)
mean = smooth(mean, 3)
err = smooth(err, 3)


colors = [u'#1f77b4', u'#ff7f0e', u'#d62728', u'#2ca02c', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
ax.plot(x[3:-3], mean[3:-3], color=colors[algo_id], label=algo_name_list[algo_id])
ax.fill_between(x[3:-3], (mean[3:-3] - err[3:-3]), (mean[3:-3] + err[3:-3]), color=colors[algo_id], alpha=.1)

ax.legend(loc='upper left', fontsize='x-large')
plt.xlabel('Episodes', fontsize='x-large')
plt.ylabel('Cumulative Rewards', fontsize='x-large')
plt.show()


# # 備份
# import numpy as np
# from matplotlib import pyplot as plt
# import os
# import math
# from RestartQ_UCB import restartq_ucb
# from LSVI_UCB_Restart import lsvi_ucb_restart
# from Q_Learning_UCB import restartq_ucb_no_restart
# from Epsilon_Greedy import q_learning_epsilon_greedy
# from Double_RestartQ_UCB import double_restartq_ucb
# import time 
# import argparse


# def smooth(y, sm=1):
#     if sm > 1:
#         ker = np.ones(sm) * 1.0 / sm
#         y = np.convolve(ker, y, "same")
#     return y


# A = 2
# H = 5
# S = 2 * H
# M = 5000
# prob_threshold = 0.98
# variation = 50
# abrupt = False
# if abrupt:
#     prefix = 'abrupt-'
# else:
#     prefix = 'gradual-'

# func_list = [restartq_ucb, lsvi_ucb_restart, restartq_ucb_no_restart, q_learning_epsilon_greedy, double_restartq_ucb]
# algo_name_list = ['RestartQ-UCB', 'LSVI-UCB-Restart', 'Q-Learning UCB', 'Epsilon-Greedy', 'Double-Restart Q-UCB']
# # 改這裡：接受 --alg 或預設 None 表示畫全部
# parser = argparse.ArgumentParser()
# parser.add_argument('--alg', type=int, default=None, help="Algorithm ID: 0~4. Leave empty to plot all.")
# args = parser.parse_args()

# x = range(M)
# fig, ax = plt.subplots()
# test_num = 5

# # 如果 args.alg 是 None，就畫全部；否則只畫指定的
# algo_ids = [args.alg] if args.alg is not None else list(range(len(func_list)))

# for algo_id in algo_ids:
#     a = []
#     for test_id in range(test_num):
#         ans = np.cumsum(func_list[algo_id](S, A, M, H, variation))
#         a.append(ans)
#     print(f"Algo {algo_name_list[algo_id]}: {len(a)} runs")
#     a = np.array(a)

#     err = []
#     mean = []

#     for j in range(M):
#         tmp = a[:, j]
#         err.append(1.96 * np.std(tmp) / np.sqrt(len(tmp)))
#         mean.append(np.mean(tmp))

#     err = np.array(err)
#     mean = np.array(mean)
#     mean = smooth(mean, 3)
#     err = smooth(err, 3)
#     colors = [u'#1f77b4', u'#ff7f0e', u'#d62728', u'#2ca02c', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
#     ax.plot(x[3:-3], mean[3:-3], color=colors[algo_id], label=algo_name_list[algo_id])
#     ax.fill_between(x[3:-3], (mean[3:-3] - err[3:-3]), (mean[3:-3] + err[3:-3]), color=colors[algo_id], alpha=.1)

# ax.legend(loc='upper left', fontsize='x-large')
# plt.xlabel('Episodes', fontsize='x-large')
# plt.ylabel('Cumulative Rewards', fontsize='x-large')
# plt.show()