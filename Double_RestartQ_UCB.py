import numpy as np
from matplotlib import pyplot as plt
import os
import math
from Combination_Lock import *



def double_restartq_ucb(S, A, M, H, variation):
    T = H * M
    W = int(np.sqrt(H * T))
    
    L = [H]
    while L[-1] < M:
        L.append(int(L[-1] * (1 + 1.0 / H)))
    for i in range(1, len(L)):
        L[i] += L[i - 1]

    J_max = int(np.log(W)) + 1
    J = []
    for j in range(J_max + 1):
        J.append(int(T * W **(j / J_max) / (S * A * H**2 * W)))
    J = [j for j in J if j != 0]
    n = len(J)
    weights = [1 / n for _ in range(n)]
    episode_rewards = np.zeros(M)
    last_d = 0
    lr = 0.2


    for w in range(int(M // W) + 1):
        if w > 0:
            weights[last_d] *= np.exp(lr * phase_reward_sum / (W * H))
            weights /= np.sum(weights)
        last_d = np.random.choice(n, p=weights)
        phase_reward_sum = 0
        D = J[last_d]
        D = int(D * W / M) + 1
        K = int(W // D) + 1

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
            for i_episode in range(K): 
                if w * W + d * K + i_episode >= M or w * W + d * K + i_episode >= (w + 1) * W:
                    break
                state = 0
                for h in range(H):
                    action = np.argmax(Q[h][state])
                    reward, next_state= transition(state, action, i_episode)
                    episode_rewards[w * W + d * K + i_episode] += reward
                    phase_reward_sum += reward

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

    return episode_rewards


