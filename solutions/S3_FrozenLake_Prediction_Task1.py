import random
import gym
import numpy as np

env = gym.make("FrozenLake-v1")
random.seed(0)

print("## Frozen Lake ##")
print("Start state:")
env.render()

action2string = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}

no_states = env.observation_space.n
no_actions = env.action_space.n

q_values = np.zeros((no_states, no_actions))
q_counter = np.zeros((no_states, no_actions))


def play_episode():
    state = env.reset()
    done = False
    r_s = []
    s_a = []
    while not done:
        action = random.randint(0, 3)
        s_a.append((state, action))
        state, reward, done, _ = env.step(action)
        r_s.append(reward)
    return s_a, r_s

def main():
    successful_episodes = 1000
    while successful_episodes > 0:
        s_a, r_s = play_episode()

        # update q-values with MC-prediction
        for i, (s,a) in enumerate(s_a):
            return_i = sum(r_s[i:])
            q_counter[s][a] += 1
            q_values[s][a] += 1/q_counter[s][a] * (return_i - q_values[s][a])

        if sum(r_s) > 0:
            print(q_values)
            successful_episodes -= 1

main()
