import random

# Agent uses Q-learning
class TicTacToeAgent:
    def __init__(self, alpha=0.1, epsilon=0.1, gamma=0.9):
        self.q_table = {}
        self.alpha = alpha # learning rate
        self.epsilon = epsilon # exploration rate
        self.gamma = gamma # discount factor

    def choose_action(self, observation):
        state_key = tuple(observation.flatten())

        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0 for action in range(len(observation.flatten()))}
        
        possible_actions = [i for i in range(len(observation.flatten())) if observation.flatten()[i] == 0]
        if random.uniform(0, 1) < self.epsilon:
            #Explore
            return random.choice(possible_actions)
        else:
            #Exploit
            action_values = {action: self.q_table[state_key].get(action, 0) for action in possible_actions}
            return max(action_values, key=action_values.get)


    def learn(self, observation, action, reward, next_observation, done):
        state_key = tuple(observation.flatten())
        next_state_key = tuple(next_observation.flatten())

        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {action: 0 for action in range(len(observation.flatten()))}
        
        max_next_q = max(self.q_table[next_state_key].values()) if not done else 0

        self.q_table[state_key][action] = self.q_table[state_key][action] + self.alpha * (reward + self.gamma * max_next_q - self.q_table[state_key][action])

