import gymnasium as gym
import torch
import torch.nn as nn

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


env = gym.make("CartPole-v1", render_mode="human")

online = DQN(128)
online.load_state_dict(torch.load("../cartpole_online_weights.pth", weights_only=True))

reset = False
s, info = env.reset(seed=5000)
step = 0

while reset == False:
    step += 1
    with torch.no_grad():
        a = torch.argmax(online(torch.tensor(s))).item()
    next_s, r, terminated, truncated, info = env.step(a)

    print(f"Game step {step} | cart position: {s[0]}, pole angle: {s[2]}")

    s = next_s
    if terminated or truncated:
        reset = True
env.close()

