import numpy as np
import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=False)
P = env.unwrapped.P

mask = {
    0 : "←",
    1: "↓",
    2: "→",
    3: "↑"
}

def value_iteration(P, gamma, theta):
    V = np.asarray([0.0] * 16, dtype=np.float64)
    converge = False
    best_acts = [0] * 16
    sweep_no = 0

    while converge == False:
        sweep_no += 1
        new_V = []
        
        for s in range(16):
            Qs = []
            for action in P[s].keys():
                Q = 0

                for action_val in P[s][action]:
                    if action_val[3]:
                        Q += action_val[0] * action_val[2] 
                    else:
                        Q += action_val[0] * (action_val[2] + gamma * V[action_val[1]])
                Qs.append(Q)

            best_acts[s] = Qs.index(max(Qs))
            new_V.append(max(Qs))

        new_V = np.array(new_V, dtype=np.float64)
        if np.max(np.abs(new_V - V)) < theta:
            converge = True
        else:
            V = new_V 
        
    return V, sweep_no, best_acts

V, sweep_num, best_acts = value_iteration(P, 0.95, 1e-8)
print(f"Value of state 0: {V[0]}; Number of sweeps: {sweep_num}")
print(*[mask[best] for best in best_acts])

    
