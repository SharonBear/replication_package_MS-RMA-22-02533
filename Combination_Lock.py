import numpy as np
from matplotlib import pyplot as plt
import os
import math


def transition(s, a, t):
    x = np.random.uniform(0, 1)
    if x < prob_threshold:
        flag = True
    else:
        flag = False

    if abrupt:
        if s == 0:
            if (a == 0) == flag:
                return (0, 2)
            else:
                return (0, 3)
        elif s == 1:
            return (0.125 / H, 1)
        elif s == 2 * H - 2:
            if (a == 0) == flag:
                if t // (M // variation) % 2 == 0:
                    return (0.25, 2 * H - 2)
                else:
                    return (1.0, 2 * H - 2)
            else:
                return (0.125 / H, 1)
        elif s == 2 * H - 1:
            if (a == 0) == flag:
                if t // (M // variation) % 2 == 0:
                    return (1.0, 2 * H - 1)
                else:
                    return (0.25, 2 * H - 1)
            else:
                return (0.125 / H, 1)
        else:
            if (a == 0) == flag:
                return (0, s + 2)
            else:
                return (0.125 / H, 1)
    else:
        variation_x = np.random.uniform(0, 1)
        if variation_x < 1.0 - 1.0 * t / M:
            variation_flag = True
        else:
            variation_flag = False
        if s == 0:
            if (a == 0):
                return (0, 2)
            else:
                return (0, 3)
        elif s == 1:
            return (0.125 / H, 1)
        elif s == 2 * H - 2:
            if (a == 0) == flag:
                return (0.25 + 0.75 * t / M, 2 * H - 2)
            else:
                return (0.125 / H, 1)
        elif s == 2 * H - 1:
            if (a == 0) == flag:
                return (1.0 - 0.75 * t / M, 2 * H - 1)
            else:
                return (0.125 / H, 1)
        else:
            if (a == 0) == flag:
                return (0, s + 2)
            else:
                return (0.125 / H, 1)


A = 2
H = 5
S = 2 * H
M = 50000
prob_threshold = 0.98
variation = 50
abrupt = False