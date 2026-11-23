import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)

WARMUP_BATCHES = 10
MAX_GRADIENT_NORM = 10.0


class QNetwork(nn.Module):

    def __init__(self, n_inputs, n_actions, hidden_units):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_inputs, hidden_units), nn.ReLU(),
            nn.Linear(hidden_units, hidden_units), nn.ReLU(),
            nn.Linear(hidden_units, n_actions))

    def forward(self, states):
        return self.layers(states)


class ReplayMemory:

    def __init__(self, capacity, n_inputs, n_actions, rng):
        self.capacity, self.rng = capacity, rng
        self.states = np.zeros((capacity, n_inputs), np.float32)
        self.actions = np.zeros(capacity, np.int64)
        self.rewards = np.zeros(capacity, np.float32)
        self.next_states = np.zeros((capacity, n_inputs), np.float32)
        self.dones = np.zeros(capacity, np.float32)
        self.next_valid = np.ones((capacity, n_actions), bool)
        self.size = self.position = 0

    def push(self, state, action, reward, next_state, done, next_valid=None):
        index = self.position
        self.states[index], self.actions[index], self.rewards[index] = state, action, reward
        self.next_states[index], self.dones[index] = next_state, float(done)
        self.next_valid[index] = True if next_valid is None else next_valid
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = self.rng.integers(0, self.size, batch_size)
        return (torch.from_numpy(self.states[indices]), torch.from_numpy(self.actions[indices]),
                torch.from_numpy(self.rewards[indices]), torch.from_numpy(self.next_states[indices]),
                torch.from_numpy(self.dones[indices]), torch.from_numpy(self.next_valid[indices]))


class DQN:

    def __init__(self, n_inputs, n_actions, discount, learning_rate, hidden_units, batch_size, memory_size,
                 target_update, train_every, seed=0):
        self.n_actions, self.discount = n_actions, discount
        self.batch_size, self.target_update, self.train_every = batch_size, target_update, train_every
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.online = QNetwork(n_inputs, n_actions, hidden_units)
        self.target = QNetwork(n_inputs, n_actions, hidden_units)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.memory = ReplayMemory(memory_size, n_inputs, n_actions, self.rng)
        self.epsilon = 0.0
        self.transitions = self.updates = 0
        self.recent_losses = []

    def q_values(self, states):
        with torch.no_grad():
            return self.online(torch.as_tensor(np.asarray(states, np.float32))).numpy()

    def act(self, states, explore, valid=None):
        states = np.atleast_2d(states)
        valid = np.ones((len(states), self.n_actions), bool) if valid is None else np.atleast_2d(valid)
        values = np.where(valid, self.q_values(states), -np.inf)
        actions = values.argmax(axis=1)
        if explore:
            for row in np.flatnonzero(self.rng.random(len(states)) < self.epsilon):
                actions[row] = self.rng.choice(np.flatnonzero(valid[row]))
        return actions

    def remember(self, state, action, reward, next_state, done, next_valid=None):
        self.memory.push(state, action, reward, next_state, done, next_valid)
        self.transitions += 1
        warm = self.memory.size >= WARMUP_BATCHES * self.batch_size
        if warm and self.transitions % self.train_every == 0:
            self._train_step()

    def _train_step(self):
        states, actions, rewards, next_states, dones, next_valid = self.memory.sample(self.batch_size)
        with torch.no_grad():
            next_values = self.target(next_states).masked_fill(~next_valid, -torch.inf).max(dim=1).values
            targets = rewards + self.discount * (1 - dones) * next_values
        predicted = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = nn.functional.smooth_l1_loss(predicted, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), MAX_GRADIENT_NORM)
        self.optimizer.step()
        self.updates += 1
        self.recent_losses.append(loss.item())
        if self.updates % self.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())

    def pop_mean_loss(self):
        losses, self.recent_losses = self.recent_losses, []
        return float(np.mean(losses)) if losses else None
