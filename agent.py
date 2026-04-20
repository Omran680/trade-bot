import numpy as np
import os
from model import DQNNetwork, PPONetwork, softmax
from replay_buffer import ReplayBuffer
from collections import deque


class DQNAgent:
    """NumPy-based DQN agent"""
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

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)

        q_values = self.model.predict(state)
        return int(np.argmax(q_values))

    def predict(self, state):
        return self.model.predict(state)

    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def replay(self, batch_size=32):
        if self.memory.size() < batch_size:
            return

        batch = self.memory.sample(batch_size)
        states = np.vstack([exp[0] for exp in batch]).astype(np.float32)
        actions = np.array([exp[1] for exp in batch], dtype=np.int32)
        rewards = np.array([exp[2] for exp in batch], dtype=np.float32)
        next_states = np.vstack([exp[3] for exp in batch]).astype(np.float32)
        dones = np.array([exp[4] for exp in batch], dtype=np.float32)

        q_values = self.model.forward(states)
        next_q_values = self.target_model.forward(next_states)
        max_next_q = np.max(next_q_values, axis=1)

        target_q = q_values.copy()
        for i in range(batch_size):
            if dones[i]:
                target_q[i, actions[i]] = rewards[i]
            else:
                target_q[i, actions[i]] = rewards[i] + self.gamma * max_next_q[i]

        error = (q_values - target_q) / batch_size

        dW3 = self.model.last_h2.T.dot(error)
        db3 = np.sum(error, axis=0)

        dh2 = error.dot(self.model.w3.T)
        dh2[self.model.last_h2 <= 0] = 0.0

        dW2 = self.model.last_h1.T.dot(dh2)
        db2 = np.sum(dh2, axis=0)

        dh1 = dh2.dot(self.model.w2.T)
        dh1[self.model.last_h1 <= 0] = 0.0

        dW1 = states.T.dot(dh1)
        db1 = np.sum(dh1, axis=0)

        self.model.w3 -= self.lr * dW3
        self.model.b3 -= self.lr * db3
        self.model.w2 -= self.lr * dW2
        self.model.b2 -= self.lr * db2
        self.model.w1 -= self.lr * dW1
        self.model.b1 -= self.lr * db1

        self.train_step += 1
        if self.train_step % self.update_target_freq == 0:
            self.target_model.set_params(self.model.get_params())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filepath):
        """Save DQN agent weights"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)

    def load(self, filepath):
        """Load DQN agent weights"""
        self.model.load(filepath)
        # Update target model as well
        self.target_model.set_params(self.model.get_params())


class PPOAgent:
    """NumPy actor-critic agent"""
    def __init__(self, state_size=10, action_size=3, lr=0.0005, gamma=0.99, gae_lambda=0.95):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.lr = lr

        self.model = PPONetwork(state_size, action_size)

        self.state_buffer = deque(maxlen=2048)
        self.action_buffer = deque(maxlen=2048)
        self.reward_buffer = deque(maxlen=2048)
        self.value_buffer = deque(maxlen=2048)
        self.logprob_buffer = deque(maxlen=2048)

    def act(self, state, deterministic=False):
        logits, value = self.model.predict(state)
        probs = softmax(logits)[0]

        if deterministic:
            action = int(np.argmax(probs))
        else:
            action = int(np.random.choice(self.action_size, p=probs))

        logprob = float(np.log(probs[action] + 1e-8))
        return action, logprob, float(value[0, 0]), probs

    def store_transition(self, state, action, reward, value, logprob):
        self.state_buffer.append(np.array(state, dtype=np.float32))
        self.action_buffer.append(int(action))
        self.reward_buffer.append(float(reward))
        self.value_buffer.append(float(value))
        self.logprob_buffer.append(float(logprob))

    def compute_gae(self, next_value):
        rewards = np.array(self.reward_buffer, dtype=np.float32)
        values = np.array(self.value_buffer, dtype=np.float32)
        trajectory_len = len(rewards)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(trajectory_len)):
            next_val = next_value if t == trajectory_len - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_val - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns, np.vstack(self.state_buffer)

    def get_value(self, state):
        _, value = self.model.predict(state)
        return float(value.reshape(-1)[0])

    def train(self, advantages, returns, states):
        actions = np.array(self.action_buffer, dtype=np.int32)
        old_logprobs = np.array(self.logprob_buffer, dtype=np.float32)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        batch_size = states.shape[0]

        for _ in range(3):
            logits, values = self.model.predict(states)
            probs = softmax(logits)
            action_indices = np.arange(batch_size)
            selected_probs = probs[action_indices, actions]
            logprobs = np.log(selected_probs + 1e-8)

            ratios = np.exp(logprobs - old_logprobs)
            policy_loss = -np.mean(ratios * advantages)
            value_loss = np.mean((returns - values.reshape(-1)) ** 2)

            dlogits = probs.copy()
            dlogits[action_indices, actions] -= 1.0
            dlogits *= (advantages[:, None] / batch_size)

            dvalues = 2.0 * (values.reshape(-1) - returns)[:, None] / batch_size

            dactor_w = self.model.last_h1.T.dot(dlogits)
            dactor_b = np.sum(dlogits, axis=0)
            dcritic_w = self.model.last_h1.T.dot(dvalues)
            dcritic_b = np.sum(dvalues, axis=0)

            dh = dlogits.dot(self.model.actor_w.T) + dvalues.dot(self.model.critic_w.T)
            dh[self.model.last_h1 <= 0] = 0.0

            dw1 = states.T.dot(dh)
            db1 = np.sum(dh, axis=0)

            self.model.actor_w -= self.lr * dactor_w
            self.model.actor_b -= self.lr * dactor_b
            self.model.critic_w -= self.lr * dcritic_w
            self.model.critic_b -= self.lr * dcritic_b
            self.model.w1 -= self.lr * dw1
            self.model.b1 -= self.lr * db1

        self.state_buffer.clear()
        self.action_buffer.clear()
        self.reward_buffer.clear()
        self.value_buffer.clear()
        self.logprob_buffer.clear()

    def save(self, filepath):
        """Save PPO agent weights"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)

    def load(self, filepath):
        """Load PPO agent weights"""
        self.model.load(filepath)


class HybridTradingAgent:
    """Hybrid PPO+DQN agent"""
    def __init__(self, state_size=10, action_size=3, dqn_weight=0.5, ppo_weight=0.5):
        self.state_size = state_size
        self.action_size = action_size
        self.dqn_weight = dqn_weight
        self.ppo_weight = ppo_weight

        self.dqn_agent = DQNAgent(state_size, action_size)
        self.ppo_agent = PPOAgent(state_size, action_size)
        self.step = 0

    def remember(self, state, action, reward, next_state, done, ppo_output):
        self.dqn_agent.remember(state, action, reward, next_state, done)
        self.ppo_agent.store_transition(state, ppo_output['action'], reward, ppo_output['value'], ppo_output['logprob'])

    def train_step(self, batch_size=32):
        self.step += 1
        self.dqn_agent.replay(batch_size)

        if self.step % 64 == 0 and len(self.ppo_agent.reward_buffer) > 0:
            next_state = np.zeros((1, self.state_size), dtype=np.float32)
            next_value = self.ppo_agent.get_value(next_state)
            advantages, returns, states = self.ppo_agent.compute_gae(next_value)
            self.ppo_agent.train(advantages, returns, states)

    def save_models(self, models_dir="./models"):
        '''Save both DQN and PPO models'''
        os.makedirs(models_dir, exist_ok=True)
        dqn_path = os.path.join(models_dir, "dqn_model.pkl")
        ppo_path = os.path.join(models_dir, "ppo_model.pkl")
        self.dqn_agent.save(dqn_path)
        self.ppo_agent.save(ppo_path)
        print(f"✓ Models saved to {models_dir}")

    def load_models(self, models_dir="./models"):
        '''Load both DQN and PPO models'''
        dqn_path = os.path.join(models_dir, "dqn_model.pkl")
        ppo_path = os.path.join(models_dir, "ppo_model.pkl")
        self.dqn_agent.load(dqn_path)
        self.ppo_agent.load(ppo_path)
        print(f"✓ Models loaded from {models_dir}")