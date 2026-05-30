from backprop_activations import Network, CostFunction, sigmoid, leakyrelu, linear, quadratic
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import copy

class Actor:
    def __init__(self, layers, activations):
        self.network = Network(layers, activations)

    def choose_action(self, state):
        action = self.network.propagate(state)
        return action

class Critic:
    def __init__(self, layers, activations):
        self.network = Network(layers, activations)

class Environment:
    def __init__(self, size, target):
        self.board = None
        self.size = size
        self.target = target

    def reset(self):
        self.board = np.array([0 for i in range(self.size * self.size)])

    def get_board1(self):
        return self.board

    def get_board2(self):
        return -self.board

    def game_ended(self):
        pass

    def check_win(self, current_player, position):
        target = np.array([current_player] * self.target)
        y, x = divmod(position, self.size)
        grid = self.board.reshape((self.size, self.size))

        rows = sliding_window_view(grid[y,:], window_shape=(self.target,))
        cols = sliding_window_view(grid[:,x], window_shape=(self.target,))

        major_diag = grid.diagonal(offset=x - y)
        if len(major_diag) >= self.target:
            major_diags = sliding_window_view(major_diag, window_shape=(self.target,))
        else:
            major_diags = np.array([[0] * self.target])

        minor_diag = grid.diagonal(offset=x - (self.size - 1 - y))
        if len(minor_diag) >= self.target:
            minor_diags = sliding_window_view(minor_diag, window_shape=(self.target,))
        else:
            minor_diags = np.array([[0] * self.target])

        return any([
            np.any(np.all(rows == target, axis=1)),
            np.any(np.all(cols == target, axis=1)),
            np.any(np.all(major_diags == target, axis=1)),
            np.any(np.all(minor_diags == target, axis=1)),
        ])

    def show(self):
        for row in self.board.reshape((self.size, self.size)):
            print("".join([" ", "X", "O"][x] for x in row))
        print()

    def play_game(self, actor, show=False):
        self.reset()
        current_player = 1
        self.states = []
        self.actions = []
        self.rewards = []
        while True:
            reward = 0
            action = actor.choose_action(current_player * self.board)
            if self.board[np.argmax(action)] != 0:
                reward -= 1
            action *= self.board == 0
            position = np.argmax(action)
            self.board[position] = current_player

            self.states.append(current_player * self.board)
            self.actions.append(position)
            self.rewards.append(reward)

            if show:
                self.show()

            if 0 not in self.board:
                self.rewards[-1] += 1
                self.rewards[-2] += 1
                break
            if self.check_win(current_player, position):
                self.rewards[-1] += 10
                self.rewards[-2] -= 10
                break
            current_player *= -1
        return np.array(self.states), np.array(self.actions), np.array(self.rewards)

def actor_loss_gradient(advantages, old_probabilities):
    def cost_func(actual, expected):
        ratio = actual / old_probabilities.T
        objective = ratio * advantages
        clipped = np.clip(ratio, 1 - clip_range, 1 + clip_range) * advantages
        return np.minimum(objective, clipped)

    def cost_func_grad(actual, expected):
        ratio = actual / old_probabilities.T
        grad = np.where(np.abs(ratio - 1) < clip_range, advantages / old_probabilities.T, 0)
        return grad
    return CostFunction(cost_func, cost_func_grad)

discount = 0.9
multistep = 0.95
clip_range = 0.2

def main():
    actor = Actor([9, 9], [sigmoid])
    critic = Critic([9, 9, 1], [leakyrelu, linear])
    environment = Environment(3, 3)
    old_actor = None

    for k in range(4000):
        states, actions, rewards = environment.play_game(actor)
        print(rewards)
        value_estimates = critic.network.propagate(states).reshape((len(states),))
        next_value_estimates = np.pad(value_estimates[1:], (0, 1))
        td = rewards + discount * next_value_estimates - value_estimates
        advantages = np.zeros_like(td)
        advantage = 0
        for i in range(len(advantages) - 1, -1, -1):
            advantage = td[i] + discount * multistep * advantage
            advantages[i] = advantage

        if old_actor is not None:
            old_probabilities = old_actor.network.propagate(states)
        else:
            old_probabilities = np.ones((len(states), 9))

        old_actor = copy.deepcopy(actor)
        cost_func = actor_loss_gradient(advantages, old_probabilities)
        actor.network.backpropagate(states, np.array([]), cost_func=cost_func)

        next_rewards = np.pad(rewards[1:], (0, 1))
        target_returns = rewards + discount * next_rewards
        print(rewards)
        critic.network.backpropagate(states, target_returns, cost_func=quadratic)

        actions = actor.network.propagate(states).T
        print(k, np.sum(cost_func.func(actions, None)) / len(actions))

    environment.play_game(actor, show=True)

if __name__ == "__main__":
    main()
