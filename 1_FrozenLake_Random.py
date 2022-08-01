import gym
import random

env = gym.make("FrozenLake-v1", is_slippery=False)

random.seed(0)
env.seed(0)

print("## Frozen Lake ##")
print("Start state:")
env.render()

no_of_actions = env.env.nA
action2string = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}

state = env.reset()
done = False

while not done:
    action = random.randint(0, no_of_actions-1)  # choose a random action
    state, reward, done, _ = env.step(action)
    print(f"\nAction:{action2string[action]}, new state:{state}, reward:{reward}")
    env.render()
