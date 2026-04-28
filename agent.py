import numpy as np
import os
from model import DQNNetwork, PPONetwork, softmax
from replay_buffer import ReplayBuffer
from collections import deque
import logging

logger = logging.getLogger(__name__)


class DQNAgent:
    """
    DQN Agent with corrected backpropagation through ReLU layers.
    
    FIX: Properly computes ReLU gradients by masking on pre-activation values,
    not post-activation values.
    """
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
        
        # For gradient monitoring
        self.last_loss = 0.0

    def act(self, state, training=True):
        """
        Epsilon-greedy action selection.
        
        Args:
            state: Current state
            training: Whether to apply epsilon-greedy (exploration)
        
        Returns:
            Action (0, 1, or 2)
        """
        if training and np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)

        q_values = self.model.predict(state)
        return int(np.argmax(q_values))

    def predict(self, state):
        """Get Q-values for a state."""
        return self.model.predict(state)

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.add(state, action, reward, next_state, done)

    def replay(self, batch_size=32):
        """
        Train on a batch from replay buffer.
        
        FIXED: ReLU gradients are now correctly computed using pre-activation masking.
        """
        if self.memory.size() < batch_size:
            return

        batch = self.memory.sample(batch_size)
        states = np.vstack([exp[0] for exp in batch]).astype(np.float32)
        actions = np.array([exp[1] for exp in batch], dtype=np.int32)
        rewards = np.array([exp[2] for exp in batch], dtype=np.float32)
        next_states = np.vstack([exp[3] for exp in batch]).astype(np.float32)
        dones = np.array([exp[4] for exp in batch], dtype=np.float32)

        # Forward pass
        q_values = self.model.forward(states)
        next_q_values = self.target_model.forward(next_states)
        max_next_q = np.max(next_q_values, axis=1)

        # Compute target Q-values (Bellman equation)
        target_q = q_values.copy()
        for i in range(batch_size):
            if dones[i]:
                target_q[i, actions[i]] = rewards[i]
            else:
                target_q[i, actions[i]] = rewards[i] + self.gamma * max_next_q[i]

        # === BACKWARD PASS (FIXED) ===
        # Compute loss gradient at output
        error = (q_values - target_q) / batch_size
        self.last_loss = np.mean((q_values - target_q) ** 2)

        # ===== LAYER 3: Output Layer (No Activation) =====
        dW3 = self.model.last_h2.T.dot(error)
        db3 = np.sum(error, axis=0)

        # ===== LAYER 2: Hidden Layer with ReLU =====
        # FIX: Gradient w.r.t h2 (pre-activation)
        dh2_pre = error.dot(self.model.w3.T)
        
        # Apply ReLU mask using PRE-ACTIVATION, not post-activation
        # h1 was pre-activation: h1 = input @ w1 + b1
        # h2_pre_activation was: h2 = relu(h1) @ w2 + b2
        dh2_pre[self.model.h2_pre <= 0] = 0.0  # ✅ Mask on pre-activation
        
        dW2 = self.model.last_h1.T.dot(dh2_pre)
        db2 = np.sum(dh2_pre, axis=0)

        # ===== LAYER 1: Hidden Layer with ReLU =====
        # FIX: Gradient w.r.t h1 (pre-activation)
        dh1_pre = dh2_pre.dot(self.model.w2.T)
        
        # Apply ReLU mask using PRE-ACTIVATION
        dh1_pre[self.model.h1_pre <= 0] = 0.0  # ✅ Mask on pre-activation
        
        dW1 = states.T.dot(dh1_pre)
        db1 = np.sum(dh1_pre, axis=0)

        # ===== UPDATE WEIGHTS =====
        self.model.w3 -= self.lr * dW3
        self.model.b3 -= self.lr * db3
        self.model.w2 -= self.lr * dW2
        self.model.b2 -= self.lr * db2
        self.model.w1 -= self.lr * dW1
        self.model.b1 -= self.lr * db1

        # Update target network periodically
        self.train_step += 1
        if self.train_step % self.update_target_freq == 0:
            self.target_model.set_params(self.model.get_params())
            logger.debug(f"Target network updated at step {self.train_step}")

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filepath):
        """Save DQN agent weights"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        logger.info(f"DQN saved to {filepath}")

    def load(self, filepath):
        """Load DQN agent weights"""
        self.model.load(filepath)
        # Update target model as well
        self.target_model.set_params(self.model.get_params())
        logger.info(f"DQN loaded from {filepath}")


class PPOAgent:
    """
    PPO Agent with corrected GAE computation.
    
    FIX: GAE now properly handles episode boundaries with done flags.
    """
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
        self.done_buffer = deque(maxlen=2048)

    def act(self, state, deterministic=False):
        """
        Sample action from policy.
        
        Args:
            state: Current state
            deterministic: If True, use argmax (no sampling)
        
        Returns:
            (action, log_prob, value, probs)
        """
        logits, value = self.model.predict(state)
        probs = softmax(logits)[0]

        if deterministic:
            action = int(np.argmax(probs))
        else:
            action = int(np.random.choice(self.action_size, p=probs))

        logprob = float(np.log(probs[action] + 1e-8))
        return action, logprob, float(value[0, 0]), probs

    def store_transition(self, state, action, reward, value, logprob, done=False):
        """
        Store transition in buffer.
        
        FIX: Now stores done flag for proper GAE computation.
        """
        self.state_buffer.append(np.array(state, dtype=np.float32))
        self.action_buffer.append(int(action))
        self.reward_buffer.append(float(reward))
        self.value_buffer.append(float(value))
        self.logprob_buffer.append(float(logprob))
        self.done_buffer.append(float(done))

    def compute_gae(self, next_value):
        """
        Compute Generalized Advantage Estimation.
        
        FIX: Properly applies (1-done) mask to break episode boundaries.
        
        Args:
            next_value: Value estimate of next state (0 if episode ended)
        
        Returns:
            (advantages, returns, states)
        """
        rewards = np.array(self.reward_buffer, dtype=np.float32)
        values = np.array(self.value_buffer, dtype=np.float32)
        dones = np.array(self.done_buffer, dtype=np.float32)
        trajectory_len = len(rewards)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(trajectory_len)):
            # Determine next value
            if t == trajectory_len - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            # ===== FIX: Apply done mask =====
            # If done[t] == 1 (episode ended), next_value should be 0
            next_val = next_val * (1 - dones[t])
            
            # TD residual
            delta = rewards[t] + self.gamma * next_val - values[t]
            
            # FIX: GAE accumulation with done mask
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns, np.vstack(self.state_buffer)

    def get_value(self, state):
        """Get value estimate for a state."""
        _, value = self.model.predict(state)
        return float(value.reshape(-1)[0])

    def train(self, advantages, returns, states):
        """
        Train actor and critic networks.
        
        Args:
            advantages: Computed advantages
            returns: Computed returns
            states: States from trajectory
        """
        actions = np.array(self.action_buffer, dtype=np.int32)
        old_logprobs = np.array(self.logprob_buffer, dtype=np.float32)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        batch_size = states.shape[0]

        # Training epochs
        for epoch in range(3):
            logits, values = self.model.forward(states)
            probs = softmax(logits)
            action_indices = np.arange(batch_size)
            selected_probs = probs[action_indices, actions]
            logprobs = np.log(selected_probs + 1e-8)

            # Policy loss with importance sampling ratio
            ratios = np.exp(logprobs - old_logprobs)
            policy_loss = -np.mean(ratios * advantages)
            
            # Value loss
            value_loss = np.mean((returns - values.reshape(-1)) ** 2)
            
            # ===== BACKWARD PASS =====
            # Gradient w.r.t logits
            dlogits = probs.copy()
            dlogits[action_indices, actions] -= 1.0
            dlogits *= (advantages[:, None] / batch_size)

            # Gradient w.r.t values
            dvalues = 2.0 * (values.reshape(-1) - returns)[:, None] / batch_size

            # Gradients for actor and critic heads
            dactor_w = self.model.last_h1.T.dot(dlogits)
            dactor_b = np.sum(dlogits, axis=0)
            dcritic_w = self.model.last_h1.T.dot(dvalues)
            dcritic_b = np.sum(dvalues, axis=0)

            # Gradient through shared layer
            dh = dlogits.dot(self.model.actor_w.T) + dvalues.dot(self.model.critic_w.T)
            dh[self.model.last_h1 <= 0] = 0.0

            dw1 = states.T.dot(dh)
            db1 = np.sum(dh, axis=0)

            # Update
            self.model.actor_w -= self.lr * dactor_w
            self.model.actor_b -= self.lr * dactor_b
            self.model.critic_w -= self.lr * dcritic_w
            self.model.critic_b -= self.lr * dcritic_b
            self.model.w1 -= self.lr * dw1
            self.model.b1 -= self.lr * db1

        # Clear buffers
        self.state_buffer.clear()
        self.action_buffer.clear()
        self.reward_buffer.clear()
        self.value_buffer.clear()
        self.logprob_buffer.clear()
        self.done_buffer.clear()

    def save(self, filepath):
        """Save PPO agent weights"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        logger.info(f"PPO saved to {filepath}")

    def load(self, filepath):
        """Load PPO agent weights"""
        self.model.load(filepath)
        logger.info(f"PPO loaded from {filepath}")


class HybridTradingAgent:
    """
    Hybrid agent combining DQN and PPO.
    
    FIX: Now correctly stores ensemble action in both buffers.
    """
    def __init__(self, state_size=10, action_size=3, dqn_weight=0.5, ppo_weight=0.5):
        self.state_size = state_size
        self.action_size = action_size
        self.dqn_weight = dqn_weight
        self.ppo_weight = ppo_weight

        self.dqn_agent = DQNAgent(state_size, action_size)
        self.ppo_agent = PPOAgent(state_size, action_size)
        self.step = 0

    def remember(self, state, action, reward, next_state, done, ppo_output):
        """
        Store transition in both agents.
        
        FIX: Uses ensemble action consistently, not PPO action.
        
        Args:
            state: Current state
            action: Ensemble action (what was actually taken)
            reward: Reward received
            next_state: Next state
            done: Episode termination
            ppo_output: PPO output dict
        """
        # DQN: stores ensemble action
        self.dqn_agent.remember(state, action, reward, next_state, done)
        
        # PPO: also stores ensemble action for consistency
        self.ppo_agent.store_transition(
            state,
            action,  # ✅ Use ensemble action
            reward,
            ppo_output['value'],
            ppo_output['logprob'],
            done
        )

    def train_step(self, batch_size=32):
        """
        Perform one training step for both agents.
        
        Args:
            batch_size: Batch size for DQN
        """
        self.step += 1
        
        # Train DQN
        self.dqn_agent.replay(batch_size)

        # Train PPO every N steps
        if self.step % 64 == 0 and len(self.ppo_agent.reward_buffer) > 0:
            next_state = np.zeros((1, self.state_size), dtype=np.float32)
            next_value = self.ppo_agent.get_value(next_state)
            advantages, returns, states = self.ppo_agent.compute_gae(next_value)
            self.ppo_agent.train(advantages, returns, states)

    def save_models(self, models_dir="./models"):
        """Save both DQN and PPO models"""
        os.makedirs(models_dir, exist_ok=True)
        dqn_path = os.path.join(models_dir, "dqn_model.pkl")
        ppo_path = os.path.join(models_dir, "ppo_model.pkl")
        self.dqn_agent.save(dqn_path)
        self.ppo_agent.save(ppo_path)
        logger.info(f"✓ Models saved to {models_dir}")

    def load_models(self, models_dir="./models"):
        """Load both DQN and PPO models"""
        dqn_path = os.path.join(models_dir, "dqn_model.pkl")
        ppo_path = os.path.join(models_dir, "ppo_model.pkl")
        
        dqn_exists = os.path.exists(dqn_path)
        ppo_exists = os.path.exists(ppo_path)
        
        if dqn_exists:
            self.dqn_agent.load(dqn_path)
        if ppo_exists:
            self.ppo_agent.load(ppo_path)
        
        if dqn_exists or ppo_exists:
            logger.info(f"✓ Models loaded from {models_dir}")
            return True
        else:
            logger.warning(f"No saved models found in {models_dir}")
            return False