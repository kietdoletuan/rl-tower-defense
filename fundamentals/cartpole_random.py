import gymnasium as gym

env = gym.make("CartPole-v1")

game_nums = 200

game_lengths = []


env.action_space.seed(1711)
for i in range(game_nums):
    game_length = 0
    reset = False

    env.reset(seed=1000 + i)

    while reset == False:
        game_length += 1
        a = env.action_space.sample()
        s_next, r, terminated, truncated, info = env.step(a)

        if terminated or truncated:
            reset = True
    game_lengths.append(game_length)
env.close()


print(f"Average game length: {sum(game_lengths) / game_nums}")
print(f"Max game length: {max(game_lengths)}")
print(f"Min game length: {min(game_lengths)}")