import numpy as np
from matplotlib import pyplot as plt
import os
import math
from Combination_Lock import *



def phi(s, a):
    ans = np.zeros(S * A)
    ans[s * A + a] = 1.0
    return ans


def lsvi_ucb_restart(S, A, M, H, variation):
    episode_rewards = np.zeros(M)
    D = variation
    K = M // D
    histS = [[0 for h in range(H)] for m in range(M)]
    histA = [[0 for h in range(H)] for m in range(M)]
    histR = np.zeros((M, H))
    Q = np.ones((M, H, S, A))
    for d in range(D):
        tau = d * K
        for i_episode in range(K):
            k = tau + i_episode
            if k % 100 == 0:
                print(k)
            state = 0
            for h in range(H - 1, -1, -1):
                lam = np.eye(S * A)
                for l in range(tau, k):
                    vec = phi(histS[l][h], histA[l][h])
                    lam += np.outer(vec, vec)
                
                tmp = np.zeros(S * A)
                for l in range(tau, k):
                    if h == H - 1:
                        tmp += phi(histS[l][h], histA[l][h]) * (histR[l][h])
                    else:
                        tmp += phi(histS[l][h], histA[l][h]) * (histR[l][h] + np.max(Q[k-1][h + 1][histS[l][h + 1]]))
                w = np.linalg.inv(lam) @ tmp
                for s in range(S):
                    for a in range(A):
                        vec = phi(s, a)
                        norm = math.sqrt(vec.T @ np.linalg.inv(lam) @ vec)
                        beta = 0.4
                        Q[k][h][s][a] = min(w.T @ vec + beta * norm, H)

            for h in range(H):
                action = np.argmax(Q[k][h][state])
                reward, next_state= transition(state, action, i_episode)
                episode_rewards[d * K + i_episode] += reward
                histS[k][h] = int(state)
                histA[k][h] = int(action)
                histR[k][h] = reward
                state = next_state

    return episode_rewards


