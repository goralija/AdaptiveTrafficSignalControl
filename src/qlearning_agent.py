import random
from config import ALPHA, GAMMA, EPSILON

class QLearningAgent:
    def __init__(self, actions, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON):
        self.q_table = {}  # key: (state, action), value: Q-value
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = actions

    def get_Q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        q_values = [self.get_Q(state, a) for a in self.actions]
        max_q = max(q_values)
        return self.actions[q_values.index(max_q)]

    def learn(self, state, action, reward, next_state):
        current_q = self.get_Q(state, action)
        max_future_q = max([self.get_Q(next_state, a) for a in self.actions])
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self.q_table[(state, action)] = new_q
