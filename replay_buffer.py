import numpy as np
import os
from collections import deque
from model import DQNNetwork, PPONetwork, softmax, relu

class ReplayBuffer:
    def __init__(self, size=5000):
        self.buffer = deque(maxlen=size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def size(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size=10, action_size=3, lr=0.001, gamma=0.95):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        self.model = DQNNetwork(state_size, action_size)
        self.target_model = self.model.copy()
        self.memory = ReplayBuffer(size=5000)
        self.update_target_freq = 1000
        self.train_step = 0

    def act(self, state, training=True):
        if training and np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        q_values = self.model.predict(state)
        return int(np.argmax(q_values))

    def replay(self, batch_size=32):
        if self.memory.size() < batch_size:
            return

        batch = self.memory.sample(batch_size)
        states = np.vstack([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.vstack([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])

        # Get Targets
        q_values = self.model.forward(states)
        next_q_values = self.target_model.forward(next_states)
        max_next_q = np.max(next_q_values, axis=1)

        target_q = q_values.copy()
        batch_indices = np.arange(batch_size)
        target_q[batch_indices, actions] = rewards + (1 - dones) * self.gamma * max_next_q

        # Backprop through 3-layer DQN
        error = (q_values - target_q) / batch_size

        # Layer 3 Gradients
        dw3 = self.model.last_h2.T.dot(error)
        db3 = np.sum(error, axis=0)

        # Layer 2 Gradients (ReLU)
        dh2 = error.dot(self.model.w3.T)
        dh2[self.model.last_h2 <= 0] = 0
        dw2 = self.model.last_h1.T.dot(dh2)
        db2 = np.sum(dh2, axis=0)

        # Layer 1 Gradients (ReLU)
        dh1 = dh2.dot(self.model.w2.T)
        dh1[self.model.last_h1 <= 0] = 0
        dw1 = states.T.dot(dh1)
        db1 = np.sum(dh1, axis=0)

        # Update Weights
        for p, g in zip([self.model.w1, self.model.b1, self.model.w2, self.model.b2, self.model.w3, self.model.b3],
                        [dw1, db1, dw2, db2, dw3, db3]):
            p -= self.lr * g

        self.train_step += 1
        if self.train_step % self.update_target_freq == 0:
            self.target_model.set_params(self.model.get_params())
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class PPOAgent:
    def __init__(self, state_size=10, action_size=3, lr=0.0005, gamma=0.99, gae_lambda=0.95):
        self.model = PPONetwork(state_size, action_size)
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.buffer = []

    def act(self, state):
        logits, value = self.model.predict(state)
        probs = softmax(logits)[0]
        action = np.random.choice(len(probs), p=probs)
        return action, np.log(probs[action] + 1e-8), value[0, 0]

    def train(self):
        if not self.buffer: return
        
        # Unpack buffer
        states = np.vstack([x[0] for x in self.buffer])
        actions = np.array([x[1] for x in self.buffer])
        old_logprobs = np.array([x[2] for x in self.buffer])
        rewards = np.array([x[3] for x in self.buffer])
        values = np.array([x[4] for x in self.buffer])
        dones = np.array([x[5] for x in self.buffer])

        # Compute GAE
        advantages = np.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            next_v = 0 if t == len(rewards)-1 else values[t+1]
            delta = rewards[t] + self.gamma * next_v * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Optimize (simplified 1-pass for clarity)
        logits, current_values = self.model.forward(states)
        probs = softmax(logits)
        
        # Policy Gradient
        curr_logprobs = np.log(probs[np.arange(len(actions)), actions] + 1e-8)
        ratio = np.exp(curr_logprobs - old_logprobs)
        
        # Gradient of Policy + Value Loss
        d_logits = probs.copy()
        d_logits[np.arange(len(actions)), actions] -= 1.0
        d_logits *= (ratio * advantages)[:, None] / len(actions)
        
        d_values = 2.0 * (current_values.flatten() - returns)[:, None] / len(actions)

        # Shared Base Backprop
        dw_actor = self.model.last_h1.T.dot(d_logits)
        db_actor = np.sum(d_logits, axis=0)
        dw_critic = self.model.last_h1.T.dot(d_values)
        db_critic = np.sum(d_values, axis=0)
        
        dh1 = d_logits.dot(self.model.actor_w.T) + d_values.dot(self.model.critic_w.T)
        dh1[self.model.last_h1 <= 0] = 0
        dw1 = states.T.dot(dh1)
        db1 = np.sum(dh1, axis=0)

        # Update Params
        for p, g in zip([self.model.w1, self.model.b1, self.model.actor_w, self.model.actor_b, self.model.critic_w, self.model.critic_b],
                        [dw1, db1, dw_actor, db_actor, dw_critic, db_critic]):
            p -= self.lr * g
        
        self.buffer = []

class HybridTradingAgent:
    def __init__(self, state_size=10, action_size=3):
        self.dqn = DQNAgent(state_size, action_size)
        self.ppo = PPOAgent(state_size, action_size)
        self.ppo_update_freq = 64

    def act(self, state, training=True):
        # Ensemble: combine DQN Q-values and PPO probabilities
        q_vals = self.dqn.model.predict(state)
        logits, ppo_v = self.ppo.model.predict(state)
        probs = softmax(logits)[0]
        
        # Logic: use DQN for value-based exploitation, PPO for policy-based exploration
        # Simple weighted sum of normalized scores
        combined = (0.5 * q_vals) + (0.5 * probs)
        action = np.argmax(combined)
        
        logprob = np.log(probs[action] + 1e-8)
        return int(action), logprob, float(ppo_v[0,0])

    def step(self, state, action, reward, next_state, done, logprob, value):
        self.dqn.memory.add(state, action, reward, next_state, done)
        self.ppo.buffer.append((state, action, reward, logprob, value, done))
        
        self.dqn.replay()
        if len(self.ppo.buffer) >= self.ppo_update_freq:
            self.ppo.train()

    def save_all(self, path="./models"):
        self.dqn.model.save(f"{path}/dqn.pkl")
        self.ppo.model.save(f"{path}/ppo.pkl")