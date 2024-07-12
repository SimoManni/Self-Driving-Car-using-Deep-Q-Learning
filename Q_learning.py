import torch
import torch.nn as nn
import torch.optim as optim

from settings import *


class ReplayBuffer(object):
    def __init__(self):
        self.mem_cntr = 0
        self.state_memory = np.zeros((MEM_SIZE, INPUT_DIMS), dtype=np.float32)
        self.new_state_memory = np.zeros((MEM_SIZE, INPUT_DIMS), dtype=np.float32)
        self.action_memory = np.zeros(MEM_SIZE, dtype=np.int8)
        self.reward_memory = np.zeros(MEM_SIZE, dtype=np.int8)
        self.terminal_memory = np.zeros(MEM_SIZE, dtype=np.bool_)

    def store_transition(self, state_array, action_array, reward_array, new_state_array, done_array):
        for state, action, reward, new_state, done in zip(state_array, action_array, reward_array, new_state_array, done_array):
            index = self.mem_cntr % MEM_SIZE
            self.state_memory[index] = state
            self.new_state_memory[index] = new_state
            self.action_memory[index] = action
            self.reward_memory[index] = reward
            self.terminal_memory[index] = 1 - done

            self.mem_cntr += 1

    def sample_buffer(self):
        max_mem = min(self.mem_cntr, MEM_SIZE)
        batch = np.random.choice(max_mem, BATCH_SIZE, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

class Agent():
    def __init__(self):
        self.policy_dqn = Brain()

        self.action_space = list(range(N_ACTIONS))
        self.memory = ReplayBuffer()

        self.epsilon = EPSILON

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def get_actions(self, state_array):
        actions = []
        for state in state_array:
            actions.append(self.choose_action(state))
        return np.array(actions)

    def choose_action(self, state):
        state = state[np.newaxis, :]

        rand = np.random.random()
        # Epsilon-greedy policy
        if rand < EPSILON:
            action = np.random.choice(self.action_space)
        else:
            actions = self.policy_dqn.predict(state)
            action = np.argmax(actions)

        return action

    def learn(self):
        if self.memory.mem_cntr > BATCH_SIZE:
            # Retrieve batch of experience
            state_batch, action_batch, reward_batch, new_state_batch, done_batch = self.memory.sample_buffer()

            # Q-values for next states
            q_next = []
            for new_state in new_state_batch:
                if not np.isnan(new_state).any():
                    q_next.append(self.policy_dqn.predict(new_state).flatten())
                else:
                    q_next.append(np.zeros(N_ACTIONS))
            q_next = np.array(q_next)
            q_eval = self.policy_dqn.predict(state_batch) # Q-values for current states

            # Bellman equation update
            max_actions = np.argmax(q_next, axis=1).astype(int)
            q_target = q_eval.copy()
            batch_index = np.arange(BATCH_SIZE, dtype=np.int32)
            q_target[batch_index, action_batch] = reward_batch + GAMMA * q_next[batch_index, max_actions] * done_batch

            # Train evaluation network
            loss = self.policy_dqn.train_batch(state_batch, np.array(q_target))
            # Epsilon decay
            self.epsilon = self.epsilon * EPSILON_DEC if self.epsilon > EPSILON_END else EPSILON_END



    def save_model(self):
        self.policy_dqn.save_model('policy_dqn.pth')

    def load_model(self):
        self.policy_dqn.load_model('policy_dqn.pth')


class Brain(nn.Module):
    def __init__(self):
        super(Brain, self).__init__()

        self.fc1 = nn.Linear(INPUT_DIMS, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, N_ACTIONS)

        self.optimizer = optim.Adam(self.parameters(), LR)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        with torch.no_grad():
            outputs = self(x).detach().numpy()
        return outputs

    def train_batch(self, batch_inputs, batch_targets):
        batch_inputs = torch.tensor(batch_inputs, dtype=torch.float32)
        batch_targets = torch.tensor(batch_targets, dtype=torch.float32)

        self.optimizer.zero_grad()

        outputs = self(batch_inputs)
        loss = self.criterion(outputs, batch_targets)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save_model(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load_model(self, filepath):
        self.load_state_dict(torch.load(filepath))
