from backprop_activations import Network, CostFunction
from ppo_test import Environment, Actor
import numpy as np
import random
import copy

def select_action(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(np.where(state == 0)[0])
    else:
        action = policy.propagate(state)
        return np.argmax(action)

learning_rate = 0.001
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 64
target_update_freq = 1000
memory_size = 10000
episodes = 1000

env = Environment(15, 5)
policy = Network([225, 225, 225])
target = copy.deepcopy(policy)
memory = []

def step_env(env, action):
    if env.board[action] != 0:
        return env.board.copy(), -2, True
    env.board[action] = 1

    if env.check_win(action):
        return env.board.copy(), 10, True
    elif 0 not in env.board:
        return env.board.copy(), 1, True

    return env.board.copy(), 0, False

def optimize_model():
    if len(memory) < batch_size:
        return

    batch = random.sample(memory, batch_size)
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
    state_batch = np.array(state_batch)
    action_batch = np.array(action_batch)
    reward_batch = np.array(reward_batch)
    next_state_batch = np.array(next_state_batch)
    done_batch = np.array(done_batch)

    indices = action_batch.reshape((len(action_batch)), 1)
    q_values = np.take_along_axis(policy.propagate(state_batch), indices, 1)
    q_values = q_values.reshape((len(q_values),))
    max_next_q_values = target.propagate(next_state_batch).max(1)
    target_q_values = reward_batch + gamma * max_next_q_values * (1 - done_batch)

    def cost_func_value(actual, expected):
        return 0.5 * np.pow(q_values - target_q_values, 2)

    def cost_func_grad(actual, expected):
        return q_values - target_q_values

    cost_func = CostFunction(cost_func_value, cost_func_grad)

    policy.backpropagate(state_batch, target_q_values, cost_func=cost_func)
    q_values = np.take_along_axis(policy.propagate(state_batch), indices, 1)
    q_values = q_values.reshape((len(q_values),))
    max_next_q_values = target.propagate(next_state_batch).max(0)[0]
    target_q_values = reward_batch + gamma * max_next_q_values * (1 - done_batch)
    # print(q_values - target_q_values)
    print(np.sum(0.5 * np.pow(q_values - target_q_values, 2)) / len(q_values))

rewards_per_episode = []
steps_done = 0

for episode in range(episodes):
    env.reset()
    state = env.board.copy()
    states = [[], []]
    actions = [[], []]

    player = 1
    while True:
        action = select_action(state, epsilon)
        if env.board[action] != 0:
            reward = -2
            break
        env.board[action] = player

        if player == 1:
            states[0].append(env.board.copy())
            actions[0].append(action)
        else:
            states[1].append(env.board.copy() * -1)
            actions[1].append(action)

        if env.check_win(player, action):
            reward = 10
            break
        elif 0 not in env.board:
            reward = 1
            break
        player *= -1

    rewards = [
        [0 for i in range(len(actions[0]))],
        [0 for i in range(len(actions[1]))]
    ]
    # 1 becomes 0 and -1 becomes 1
    winner = (1 - env.check_win(player, action)) // 2

    current_reward = reward
    for i in reversed(range(len(actions[winner]))):
        rewards[winner][i] = current_reward
        current_reward *= 0.7

    current_reward = -reward
    for i in reversed(range(len(actions[1 - winner]))):
        rewards[1 - winner][i] = current_reward
        current_reward *= 0.7

    next_states = [[], []]
    for i in range(len(states[0])):
        next_idx = min(i + 1, len(states[0]) - 1)
        next_states[0].append(states[0][next_idx])
    for i in range(len(states[1])):
        next_idx = min(i + 1, len(states[1]) - 1)
        next_states[1].append(states[1][next_idx])

    terminal_states = [
        [False for i in range(len(actions[0]) - 1)] + [True],
        [False for i in range(len(actions[1]) - 1)] + [True],
    ]

    memory.extend(zip(
        states[0] + states[1],
        actions[0] + actions[1],
        rewards[0] + rewards[1],
        next_states[0] + next_states[1],
        terminal_states[0] + terminal_states[1]
    ))

    optimize_model()
    if steps_done % target_update_freq > 0:
        steps_done %= target_update_freq
        target = copy.deepcopy(policy)
    steps_done += len(states[0]) + len(states[1])

    # episode_reward = 0
    # done = False

    # while not done:
    #     # Select action
    #     action = select_action(state, epsilon)
    #     next_state, reward, done = step_env(env, action)
    #     memory.append((state, action, reward, next_state, done))
    #     state = next_state
    #     episode_reward += reward
    #     optimize_model()
    #     if steps_done % target_update_freq == 0:
    #         target = copy.deepcopy(policy)

    #     steps_done += 1

    # epsilon = max(epsilon_min, epsilon_decay * epsilon)
    # rewards_per_episode.append(episode_reward)
    print(episode)

actor = Actor.__new__(Actor)
actor.network = policy
env.play_game(actor, True)
