import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from settings import *


class ReplayBuffer(object):
    def __init__(self, discrete=False):
        self.mem_cntr = 0
        self.discrete = discrete
        self.state_memory = np.zeros((MEM_SIZE, INPUT_DIMS))
        self.new_state_memory = np.zeros((MEM_SIZE, INPUT_DIMS))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((MEM_SIZE, N_ACTIONS), dtype=dtype)
        self.reward_memory = np.zeros(MEM_SIZE)
        self.terminal_memory = np.zeros(MEM_SIZE, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % MEM_SIZE
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        # Store one hot encoding of actions
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self):
        max_mem = min(self.mem_cntr, MEM_SIZE)
        batch = np.random.choice(max_mem, BATCH_SIZE)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

class Agent():
    def __init__(self):
        self.brain_eval = Brain()
        self.brain_target = Brain()

        self.action_space = list(range(N_ACTIONS))
        self.memory = ReplayBuffer(discrete=True)

        self.epsilon = EPSILON

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):
        state = np.array(state)
        state = state[np.newaxis, :]

        rand = np.random.random()
        # Epsilon-greedy policy
        if rand < EPSILON:
            action = np.random.choice(self.action_space)
        else:
            actions = self.brain_eval(state)
            action = np.argmax(actions)

        return action

    def learn(self):
        if self.memory.mem_cntr > BATCH_SIZE:
            # Retrieve batch of experience
            state, action, reward, new_state, done = self.memory.sample_buffer()

            # Convert selected actions into indices
            action_values = np.array(self.action_space, dtype=np.int8)
            action_indices = np.dot(action, action_values)

            # Compute q values for current and next state
            q_next = self.brain_target(new_state).detach().numpy()
            q_eval = self.brain_eval(new_state).detach().numpy()
            q_pred = self.brain_eval(state).detach().numpy()

            # Bellman equation update
            max_actions = np.argmax(q_eval, axis=1)
            q_target = q_pred
            batch_index = np.arange(BATCH_SIZE, dtype=np.int32)
            q_target[batch_index, action_indices] = reward + GAMMA * q_next[batch_index, max_actions.astype(int)] * done

            # Train evaluation network
            _ = self.brain_eval.train_batch(state, np.array(q_target))

            # Epsilon decay
            self.epsilon = self.epsilon * EPSILON_DEC if self.epsilon > EPSILON_END else EPSILON_END

            # Update parameters of target network
            if self.epsilon == 0.0:
                self.update_network_parameters()

    def update_network_parameters(self):
        self.brain_target.copy_weights(self.brain_eval)

    def save_model(self):
        self.brain_eval.save_model('brain_eval.pth')
        self.brain_target.save_model('brain_target.pth')

    def load_model(self):
        self.brain_eval.load_model('brain_eval.pth')
        self.brain_target.load_model('brain_target.pth')


class Brain(nn.Module):
    def __init__(self):
        super(Brain, self).__init__()

        self.fc1 = nn.Linear(INPUT_DIMS, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, N_ACTIONS)
        self.softmax = nn.Softmax(dim=1)

        self.optimizer = optim.Adam(self.parameters(), LR)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

    def train_batch(self, batch_inputs, batch_targets):
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