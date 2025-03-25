import numpy as np
from matplotlib import pyplot as plt
import os
import math
from Combination_Lock import *



def restartq_ucb(S, A, M, H, variation):
    L = [H]
    while L[-1] < M:
        L.append(int(L[-1] * (1 + 1.0 / H)))
    for i in range(1, len(L)):
        L[i] += L[i - 1]
    episode_rewards = np.zeros(M)
    D = variation
    K = M // D
    for d in range(D):
        Q = np.ones((H, S, A))
        for h in range(H):
            Q[h, :, :] = (H - h) * np.ones((S, A))

        V = np.ones((H, S))
        for h in range(H):
            V[h, :] = (H - h) * np.ones(S)

        N = np.zeros((H, S, A))
        N_check = np.zeros((H, S, A))
        r_check = np.zeros((H, S, A))
        v_check = np.zeros((H, S, A))
        optimal_reward_per_episode = H  # 每一步都得到 1 分的總和
        regrets = np.zeros(M)
        for i_episode in range(K): 
            state = 0
            for h in range(H):
                action = np.argmax(Q[h][state])
                reward, next_state= transition(state, action, i_episode)
                episode_rewards[d * K + i_episode] += reward

                r_check[h][state][action] += reward
                if h != H - 1:
                    v_check[h][state][action] += V[h + 1][next_state]
                N[h][state][action] += 1
                N_check[h][state][action] += 1

                if N[h][state][action] in L:
                    bonus = 0.05 * np.sqrt(H**2 / N_check[h][state][action])
                    Q[h][state][action] = min(Q[h][state][action], r_check[h][state][action] / N_check[h][state][action] + v_check[h][state][action] / N_check[h][state][action] + bonus)
                    V[h][state] = np.max(Q[h][state])
                    N_check[h][state][action] = 0
                    r_check[h][state][action] = 0.0
                    v_check[h][state][action] = 0.0
                state = next_state
            # 計算 regret
            regrets[d * K + i_episode] = optimal_reward_per_episode - episode_rewards[d * K + i_episode]
    return episode_rewards,regrets


