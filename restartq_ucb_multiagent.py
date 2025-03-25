import numpy as np
from combination_lock_multiagent import transition, S, A, H, M

def restartq_ucb_multiagent(M, H):
    L = [H]
    while L[-1] < M:
        L.append(int(L[-1] * (1 + 1.0 / H)))
    for i in range(1, len(L)):
        L[i] += L[i - 1]

    episode_rewards = np.zeros(M)
    regrets = np.zeros(M)
    optimal_reward_per_episode = 1.0  # 只有在最終成功才拿到

    Q_A = np.ones((H, S * S, A)) * H
    Q_B = np.ones((H, S * S, A)) * H
    V_A = np.ones((H, S * S)) * H
    V_B = np.ones((H, S * S)) * H

    N_A = np.zeros((H, S * S, A))
    N_B = np.zeros((H, S * S, A))
    N_check_A = np.zeros((H, S * S, A))
    N_check_B = np.zeros((H, S * S, A))
    r_check_A = np.zeros((H, S * S, A))
    r_check_B = np.zeros((H, S * S, A))
    v_check_A = np.zeros((H, S * S, A))
    v_check_B = np.zeros((H, S * S, A))

    for ep in range(M):
        s_x, s_y = 0, 0
        for h in range(H):
            state = s_x * S + s_y
            a1 = np.argmax(Q_A[h][state])
            a2 = np.argmax(Q_B[h][state])

            r, next_s_x, next_s_y = transition(s_x, s_y, a1, a2, ep)
            episode_rewards[ep] += r

            if h != H - 1:
                next_state = next_s_x * S + next_s_y
                v_check_A[h][state][a1] += V_A[h + 1][next_state]
                v_check_B[h][state][a2] += V_B[h + 1][next_state]
            r_check_A[h][state][a1] += r
            r_check_B[h][state][a2] += r
            N_A[h][state][a1] += 1
            N_B[h][state][a2] += 1
            N_check_A[h][state][a1] += 1
            N_check_B[h][state][a2] += 1

            if N_A[h][state][a1] in L:
                bonus = 0.05 * np.sqrt(H ** 2 / N_check_A[h][state][a1])
                Q_A[h][state][a1] = min(
                    Q_A[h][state][a1],
                    r_check_A[h][state][a1] / N_check_A[h][state][a1] +
                    v_check_A[h][state][a1] / N_check_A[h][state][a1] + bonus
                )
                V_A[h][state] = np.max(Q_A[h][state])
                N_check_A[h][state][a1] = 0
                r_check_A[h][state][a1] = 0.0
                v_check_A[h][state][a1] = 0.0

            if N_B[h][state][a2] in L:
                bonus = 0.05 * np.sqrt(H ** 2 / N_check_B[h][state][a2])
                Q_B[h][state][a2] = min(
                    Q_B[h][state][a2],
                    r_check_B[h][state][a2] / N_check_B[h][state][a2] +
                    v_check_B[h][state][a2] / N_check_B[h][state][a2] + bonus
                )
                V_B[h][state] = np.max(Q_B[h][state])
                N_check_B[h][state][a2] = 0
                r_check_B[h][state][a2] = 0.0
                v_check_B[h][state][a2] = 0.0

            s_x, s_y = next_s_x, next_s_y

        regrets[ep] = optimal_reward_per_episode - episode_rewards[ep]

    return episode_rewards, regrets