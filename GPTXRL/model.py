import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as  np

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, hidden_dims1, hidden_dims2, n_actions) -> None:
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.hidden_dims1 = hidden_dims1
        self.hidden_dims2 = hidden_dims2
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.hidden_dims1)
        self.fc2 = nn.Linear(self.hidden_dims1, self.hidden_dims2)
        self.fc3 = nn.Linear(self.hidden_dims2, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions

class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, n_actions, batch_size,  max_mem_size=100000, eps_min=0.01, eps_dec=5e-4, hidden_size=512):
        # Variables.
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.lr = lr
        # Layer variables.
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.action_space = [i for i in range(self.n_actions)]
        self.max_mem_size = max_mem_size
        # Eps.
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        # Counters.
        self.mem_counter = 0
        # Dims (hidden layers).
        self.hidden_dims1 = hidden_size
        self.hidden_dims2 = hidden_size
        # Model.
        self.Q_eval = DeepQNetwork(self.lr, n_actions=self.n_actions, input_dims=self.input_dims, hidden_dims1=self.hidden_dims1, hidden_dims2=self.hidden_dims2)
        # States.
        self.state_memory = np.zeros((self.max_mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.max_mem_size, *input_dims), dtype=np.float32)
        # Memory.
        self.action_memory = np.zeros(self.max_mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.max_mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.max_mem_size, dtype=np.bool_)
    
    def store_transition(self, state, action, reward, state_, done):
        # Gets the index with modulo math.
        index = self.mem_counter % self.max_mem_size
        # Adjust states.
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        # Adds one to the memory counter.
        self.mem_counter += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = torch.tensor(np.array(observation)).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def save_model(self, path):
        torch.save(self.Q_eval.state_dict(), path)

    def load_model(self, path):
        self.Q_eval.load_state_dict(torch.load(path))
        self.Q_eval.eval()

    def learn(self):
        if self.mem_counter < self.batch_size:
            return
        
        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_counter, self.max_mem_size)
        batch = np.random.choice(self.max_mem_size, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = torch.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        return loss.item()
