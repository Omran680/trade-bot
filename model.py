import numpy as np
import pickle
import os
import logging

logger = logging.getLogger(__name__)


def relu(x):
    """ReLU activation function."""
    return np.maximum(0.0, x)


def softmax(x):
    """Softmax activation function."""
    x = np.atleast_2d(x).astype(np.float32)
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / (np.sum(exp_x, axis=1, keepdims=True) + 1e-8)


class DQNNetwork:
    """
    NumPy-based DQN network with corrected backpropagation.
    
    FIX: Now stores both pre-activation and post-activation for correct ReLU gradient computation.
    
    Architecture:
    input -> linear -> ReLU -> linear -> ReLU -> linear -> output
    """
    def __init__(self, state_size, action_size, hidden_dim=128, seed=None):
        rng = np.random.RandomState(seed)
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_dim = hidden_dim

        # Xavier initialization
        self.w1 = rng.randn(state_size, hidden_dim).astype(np.float32) * np.sqrt(2.0 / state_size)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)

        self.w2 = rng.randn(hidden_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(hidden_dim, dtype=np.float32)

        self.w3 = rng.randn(hidden_dim, action_size).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.b3 = np.zeros(action_size, dtype=np.float32)

        # ===== FIX: Store both pre and post activation =====
        self.last_input = None
        
        # Pre-activation values (needed for ReLU gradient)
        self.h1_pre = None
        self.h2_pre = None
        
        # Post-activation values (needed for weight gradients)
        self.last_h1 = None
        self.last_h2 = None

    def forward(self, x):
        """
        Forward pass through network.
        
        Stores both pre and post-activation for backprop.
        """
        x = np.atleast_2d(x).astype(np.float32)
        self.last_input = x
        
        # Layer 1: input -> h1
        self.h1_pre = x.dot(self.w1) + self.b1  # Pre-activation
        self.last_h1 = relu(self.h1_pre)        # Post-activation
        
        # Layer 2: h1 -> h2
        self.h2_pre = self.last_h1.dot(self.w2) + self.b2  # Pre-activation
        self.last_h2 = relu(self.h2_pre)                   # Post-activation
        
        # Layer 3: h2 -> output (no activation)
        output = self.last_h2.dot(self.w3) + self.b3
        
        return output

    def predict(self, state):
        """
        Get output for a state (inference mode).
        
        Args:
            state: State vector or batch of states
        
        Returns:
            Q-values vector or batch
        """
        q_values = self.forward(state)
        return q_values.reshape(-1) if q_values.ndim == 2 and q_values.shape[0] == 1 else q_values

    def get_params(self):
        """Get all network parameters."""
        return {
            'w1': self.w1.copy(),
            'b1': self.b1.copy(),
            'w2': self.w2.copy(),
            'b2': self.b2.copy(),
            'w3': self.w3.copy(),
            'b3': self.b3.copy()
        }

    def set_params(self, params):
        """Set all network parameters."""
        self.w1 = params['w1'].copy()
        self.b1 = params['b1'].copy()
        self.w2 = params['w2'].copy()
        self.b2 = params['b2'].copy()
        self.w3 = params['w3'].copy()
        self.b3 = params['b3'].copy()

    def copy(self):
        """Create a deep copy of the network."""
        clone = DQNNetwork(self.state_size, self.action_size, self.hidden_dim)
        clone.set_params(self.get_params())
        return clone

    def save(self, filepath):
        """Save network weights to file."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        params = self.get_params()
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)
        logger.info(f"DQN Network saved to {filepath}")

    def load(self, filepath):
        """Load network weights from file."""
        if not os.path.exists(filepath):
            logger.warning(f"Warning: {filepath} not found. Using initialized weights.")
            return
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        self.set_params(params)
        logger.info(f"DQN Network loaded from {filepath}")


class PPONetwork:
    """
    NumPy-based actor-critic network for PPO.
    
    Architecture:
    - Shared base: input -> linear -> ReLU
    - Actor head: h1 -> linear -> output (logits)
    - Critic head: h1 -> linear -> output (value)
    """
    def __init__(self, state_size, action_size, hidden_dim=128, seed=None):
        rng = np.random.RandomState(seed)
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_dim = hidden_dim

        # Shared base layer
        self.w1 = rng.randn(state_size, hidden_dim).astype(np.float32) * np.sqrt(2.0 / state_size)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)

        # Actor head
        self.actor_w = rng.randn(hidden_dim, action_size).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.actor_b = np.zeros(action_size, dtype=np.float32)

        # Critic head
        self.critic_w = rng.randn(hidden_dim, 1).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.critic_b = np.zeros(1, dtype=np.float32)

        # FIX: Store pre and post activation
        self.last_input = None
        self.h1_pre = None
        self.last_h1 = None

    def forward(self, x):
        """
        Forward pass through actor-critic network.
        
        Returns:
            (logits, values): Actor logits and critic values
        """
        x = np.atleast_2d(x).astype(np.float32)
        self.last_input = x
        
        # Shared base
        self.h1_pre = x.dot(self.w1) + self.b1
        self.last_h1 = relu(self.h1_pre)

        # Actor and critic heads
        logits = self.last_h1.dot(self.actor_w) + self.actor_b
        values = self.last_h1.dot(self.critic_w) + self.critic_b
        
        return logits, values

    def predict(self, state):
        """Get logits and values for a state."""
        return self.forward(state)

    def get_params(self):
        """Get all network parameters."""
        return {
            'w1': self.w1.copy(),
            'b1': self.b1.copy(),
            'actor_w': self.actor_w.copy(),
            'actor_b': self.actor_b.copy(),
            'critic_w': self.critic_w.copy(),
            'critic_b': self.critic_b.copy()
        }

    def set_params(self, params):
        """Set all network parameters."""
        self.w1 = params['w1'].copy()
        self.b1 = params['b1'].copy()
        self.actor_w = params['actor_w'].copy()
        self.actor_b = params['actor_b'].copy()
        self.critic_w = params['critic_w'].copy()
        self.critic_b = params['critic_b'].copy()

    def save(self, filepath):
        """Save network weights to file."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        params = self.get_params()
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)
        logger.info(f"PPO Network saved to {filepath}")

    def load(self, filepath):
        """Load network weights from file."""
        if not os.path.exists(filepath):
            logger.warning(f"Warning: {filepath} not found. Using initialized weights.")
            return
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        self.set_params(params)
        logger.info(f"PPO Network loaded from {filepath}")