import numpy as np
from matplotlib import pyplot as plt
import os
import math
from Combination_Lock import *



def epsilon_greedy_exploration(Q, epsilon, num_actions):
    def policy_exp(state):
        probs = np.ones(num_actions, dtype=float) * epsilon / num_actions
        best_action = np.argmax(Q[state])
        probs[best_action] += (1.0 - epsilon)
        return probs
    return policy_exp




def q_learning_epsilon_greedy(S, A, M, H, variation, epsilon=0.05):
    D = variation
    K = M // D
    episode_rewards = np.zeros(M)
    for d in range(D):
        Q = np.zeros((S, A))
        policy = np.ones(S)
        V = np.zeros((M, S))
        for i_episode in range(K): 
            greedy_probs = epsilon_greedy_exploration(Q, epsilon, A)
            N = np.zeros((S, A))
            state = 0
            for h in range(H):
                # epsilon greedy exploration
                action_probs = greedy_probs(state)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                reward, next_state = transition(state, action, 1)
                episode_rewards[d * K + i_episode] += reward
                N[state, action] += 1
                alpha = 1 / (h + 1)**0.85
                best_next_action = np.argmax(Q[next_state])    
                td_target = reward + Q[next_state][best_next_action]
                td_delta = td_target - Q[state][action]
                Q[state][action] += alpha * td_delta
                state = next_state
            V[i_episode,:] = Q.max(axis=1)
            policy = Q.argmax(axis=1)
    return episode_rewards

