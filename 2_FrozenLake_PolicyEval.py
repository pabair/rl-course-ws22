import gym
import random

env = gym.make("FrozenLake-v1", is_slippery=True)
discount = 1

random.seed(0)
env.seed(0)

actions = range(0, env.env.nA)
states = range(0, env.env.nS)
tp_matrix = env.env.P

action2string = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}
env.render()

# put your solution here





