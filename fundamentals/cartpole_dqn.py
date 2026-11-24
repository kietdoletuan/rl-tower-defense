import torch
import gymnasium as gym
import torch.nn as nn
import replay_buffer
import random
import numpy as np

torch.set_num_threads(1)

env = gym.make("CartPole-v1")

class DQN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(4, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 2)
        )

    def forward(self, x):
        return self.layer(x)

gamma = 0.99
lr = 0.0005
steps = 100000
seed = 0
test_games = 20

random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

online = DQN(128)
target = DQN(128)

buffer = replay_buffer.buffer(50000)
optimizer = torch.optim.Adam(online.parameters(), lr=lr)
criterion = nn.HuberLoss()

s, info = env.reset(seed=seed)
env.action_space.seed(seed)

val_env = gym.make("CartPole-v1")
s_val, info_val = val_env.reset(seed=7)

game_num = 0
val_gamels = []
val_gamel = 0

for i in range(steps):
    val_gamel += 1
    if i % 500 == 0:
        target.load_state_dict(online.state_dict())

    if i % 10000 == 0 and len(val_gamels) >= 20:
        ls = np.asarray(val_gamels)
        print(f"========STEP {i}========")
        print(f"Number of games so far: {game_num}")
        print(f"Mean of last 20 games: {np.sum(ls[-20:]) / 20}")

        with torch.no_grad():
            val_outs = online(torch.tensor(s_val))
            print(f"Max Q: {torch.max(val_outs).item()}")



    epsilon = max([0.05, 1 - 0.000095*(i)])

    if random.random() < epsilon:
        a = env.action_space.sample()
    else:
        with torch.no_grad():
            output = online(torch.tensor(s))
        a = torch.argmax(output).item()

    s_next, r, terminated, truncated, info = env.step(a)

    buffer.push(s, a, r, s_next, terminated)

    if len(buffer) >= 1000:
        samples = buffer.sample(64)

        states, actions, rewards, next_states, terminateds = samples

        states = np.asarray(states)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards)
        next_states = np.asarray(next_states)
        terminateds = np.asarray(terminateds)

        states, next_states = torch.tensor(states, dtype=torch.float32),torch.tensor(next_states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards, terminateds = torch.tensor(rewards, dtype=torch.float32),torch.tensor(terminateds, dtype=torch.float32)

        
        with torch.no_grad():
            target_q_out = target(next_states) 
            max_q_targets = torch.max(target_q_out, dim=1).values
            y = rewards + gamma * (1-terminateds) * max_q_targets

        outputs = online(states)
        outs = torch.gather(outputs, dim=1, index=actions.unsqueeze(1)).squeeze(1)  
        loss = criterion(outs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    s = s_next
    if terminated or truncated:
        s, info = env.reset()
        game_num += 1
        val_gamels.append(val_gamel)
        val_gamel = 0

game_lengths = []

torch.save(online.state_dict(), "cartpole_online_weights.pth")

for i in range(test_games):

    step = 0

    game_length = 0
    reset = False
    s, info = env.reset(seed=5000 + i)
    while reset == False:
        step += 1
        
        with torch.no_grad():
            a = torch.argmax(online(torch.tensor(s))).item()
        
        next_s, r, terminated, truncated, info = env.step(a)

        s = next_s
        if terminated or truncated:
            reset = True

    game_lengths.append(step)

print(f"Mean test game length: {sum(game_lengths) / len(game_lengths)}")
print(f"Min game length: {min(game_lengths)}")


