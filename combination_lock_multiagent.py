# combination_lock_multiagent.py
import numpy as np

# 環境參數
S = 5  # 每個 agent 控制的狀態空間大小
A = 2  # 每個 agent 可選的動作 0 or 1
H = 5
M = 5000
prob_threshold = 0.98
variation = 50
abrupt = False

# state 由 (s_x, s_y) 組成，共 SxS 個狀態

def transition(s_x, s_y, a1, a2, t):
    a = (a1 << 1) | a2  # 組合成單一動作 (0~3)
    x = np.random.uniform(0, 1)
    flag = x < prob_threshold

    # 是否為正確組合
    correct = (a1 == flag) and (a2 != flag)

    # 狀態更新：正確則往右下，錯誤則重置
    if correct:
        next_s_x = min(s_x + 1, S - 1)
        next_s_y = min(s_y + 1, S - 1)
    else:
        next_s_x = 0
        next_s_y = 0

    # 只有走到右下角才給 reward
    is_final = (s_x == S - 1) and (s_y == S - 1)
    reward = 1.0 if is_final else 0.0

    return reward, next_s_x, next_s_y