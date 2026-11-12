import gymnasium as gym
import numpy as np
import random

env = gym.make("FrozenLake-v1", is_slippery=False)

Q = np.zeros((16, 4))

episodes = 2000
alpha = 0.1
gamma = 0.95

mask = {
    0 : "←",
    1: "↓",
    2: "→",
    3: "↑"
}

def choose_action(Q, s, epsilon, env):
    if random.random() < epsilon:

        action = env.action_space.sample()

        return action

    top_a = np.max(Q[s])
    tied = []

    for i in range(len(Q[s])):
        if Q[s][i] == top_a:
            tied.append(i)

    action = random.choice(tied)

    return action

for e in range(episodes):
    s, info = env.reset()
    reset = False
    a = choose_action(Q, s, 0.1, env)

    while reset == False:

        s_next, r, terminated, truncated, info = env.step(a)
        a_next = choose_action(Q, s_next, 0.1, env)


        old = Q[s][a]

        if terminated:
            y = r
        else:
            y = r + gamma * Q[s_next][a_next] 

        delta = y - old

        Q[s][a] = old + alpha * delta

        s = s_next
        a = a_next

        if terminated or truncated:
            reset = True

print(f"max Q[0]: {np.max(Q[0])}")

wins = 0

for i in range(1000):
    s, info = env.reset()
    reset = False
    a = choose_action(Q, s, 0, env)

    while reset == False:
        
        s_next, r, terminated, truncated, info = env.step(a)
        a_next = choose_action(Q, s_next, 0, env)

        s = s_next
        a = a_next

        if r == 1:
            wins += 1

        if terminated or truncated:
            reset = True

print(f"Non slippery: {(wins)/1000}")


Q = np.zeros((16, 4))

wins = 0

env2 = gym.make("FrozenLake-v1", is_slippery=True)

for e in range(episodes):
    s, info = env2.reset()
    reset = False
    a = choose_action(Q, s, 0.1, env2)

    while reset == False:

        s_next, r, terminated, truncated, info = env2.step(a)
        a_next = choose_action(Q, s_next, 0.1, env2)


        old = Q[s][a]

        if terminated:
            y = r
        else:
            y = r + gamma * Q[s_next][a_next] 

        delta = y - old

        Q[s][a] = old + alpha * delta

        s = s_next
        a = a_next

        if terminated or truncated:
            reset = True

print(f"max Q[0]: {np.max(Q[0])}")

for i in range(1000):
    s, info = env2.reset()
    reset = False
    a = choose_action(Q, s, 0, env2)

    while reset == False:
        
        s_next, r, terminated, truncated, info = env2.step(a)
        a_next = choose_action(Q, s_next, 0, env2)

        s = s_next
        a = a_next

        if r == 1:
            wins += 1

        if terminated or truncated:
            reset = True

print(f"Slippery: {(wins)/1000}")




