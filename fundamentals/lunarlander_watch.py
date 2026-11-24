import gymnasium as gym
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(8, dim),
            nn.ReLU(),
            nn.Linear(dim, dim*2),
            nn.ReLU(),
            nn.Linear(dim*2, dim),
            nn.ReLU(),
            nn.Linear(dim, 4)
        )

    def forward(self, x):
        return self.layer(x)


env = gym.make("LunarLander-v3", render_mode="human")

online = DQN(128)
online.load_state_dict(torch.load("lunarlander_online_weights.pth", weights_only=True))

games = 20

for i in range(games):
    reset = False
    s, info = env.reset(seed=(5000 + i))
    while reset == False:
        with torch.no_grad():
            a = torch.argmax(online(torch.tensor(s))).item()
        next_s, r, terminated, truncated, info = env.step(a)

        s = next_s
        if terminated or truncated:
            reset = True
env.close()

