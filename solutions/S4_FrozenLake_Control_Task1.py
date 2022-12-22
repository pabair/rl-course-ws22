import random
import gym
import matplotlib.pyplot as plt
import numpy as np

env = gym.make("FrozenLake-v1")
random.seed(0)
np.random.seed(0)
env.seed(0)

print("## Frozen Lake ##")
print("Start state:")
env.render()

no_states = env.observation_space.n
no_actions = env.action_space.n


def play_episode(q_values, epsilon):

    state = env.reset()
    done = False
    r_s = []
    s_a = []
    while not done:
        action = choose_action(q_values, state, epsilon)

        s_a.append((state, action))
        state, reward, done, _ = env.step(action)
        r_s.append(reward)
    return s_a, r_s


def choose_action(q_values, state, epsilon):
    if random.random() > epsilon:
        # note: there can be more than one best action, e.g. [0, 0, 0, 0]
        # problem: np.argmax(q_values[state]) will always return the first max element
        # better: find all max elements and select one randomly
        max_indices = [i for i, v in enumerate(q_values[state]) if v == max(q_values[state])]
        return random.choice(max_indices)
    else:
        return random.randint(0, 3)


def main():
    no_episodes = 1000
    epsilons = [0.01, 0.1, 0.5, 1.0]

    plot_data = []
    for e in epsilons:

        q_values = np.zeros((no_states, no_actions))
        q_counter = np.zeros((no_states, no_actions))

        rewards = []
        for j in range(0, no_episodes):
            s_a, r = play_episode(q_values, epsilon=e)
            rewards.append(sum(r))

            # update q-values with MC-prediction
            for i, (s, a) in enumerate(s_a):
                return_i = sum(r[i:])
                q_counter[s][a] += 1
                q_values[s][a] += 1/q_counter[s][a] * (return_i - q_values[s][a])

        plot_data.append(np.cumsum(rewards))

    plt.figure()
    plt.xlabel("No. of episodes")
    plt.ylabel("Total reward")
    for i, eps in enumerate(epsilons):
        plt.plot(range(0, no_episodes), plot_data[i], label="e=" + str(eps))
    plt.legend()
    plt.show()


main()
