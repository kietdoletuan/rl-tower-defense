import gymnasium as gym
import numpy as np
import random

env = gym.make("FrozenLake-v1", is_slippery=True)

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

def choose_action(Q, s, epsilon):
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
    while reset == False:
        a = choose_action(Q, s, 0.1)
        s_next, r, terminated, truncated, info = env.step(a)

        old = Q[s][a]

        if terminated:
            y = r
        else:
            y = r + gamma * np.max(Q[s_next])

        delta = y - old

        Q[s][a] = old + alpha * delta

        s = s_next

        if terminated or truncated:
            reset = True
    

print(np.max(Q[0]))

s, info = env.reset()
reset = False


while reset == False:
    gameboard = np.zeros((16))
    gameboard[s] = 1
    print(gameboard.reshape(4, 4))
    a = choose_action(Q, s, 0)
    s_next, r, terminated, truncated, info = env.step(a)

    print(f"Action: {mask[a]}: Landed: {s_next}")

    s = s_next
    if terminated or truncated:
        reset = True