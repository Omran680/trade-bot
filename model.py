import numpy as np
import pickle
import os


def relu(x):
    return np.maximum(0.0, x)


def softmax(x):
    x = np.atleast_2d(x).astype(np.float32)
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / (np.sum(exp_x, axis=1, keepdims=True) + 1e-8)


class DQNNetwork:
    """Simple NumPy-based DQN network"""
    def __init__(self, state_size, action_size, hidden_dim=128, seed=None):
        rng = np.random.RandomState(seed)
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_dim = hidden_dim

        self.w1 = rng.randn(state_size, hidden_dim).astype(np.float32) * np.sqrt(2.0 / state_size)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)

        self.w2 = rng.randn(hidden_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(hidden_dim, dtype=np.float32)

        self.w3 = rng.randn(hidden_dim, action_size).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.b3 = np.zeros(action_size, dtype=np.float32)

        self.last_input = None
        self.last_h1 = None
        self.last_h2 = None

    def forward(self, x):
        x = np.atleast_2d(x).astype(np.float32)
        self.last_input = x
        self.last_h1 = relu(x.dot(self.w1) + self.b1)
        self.last_h2 = relu(self.last_h1.dot(self.w2) + self.b2)
        return self.last_h2.dot(self.w3) + self.b3

    def predict(self, state):
        q_values = self.forward(state)
        return q_values.reshape(-1) if q_values.ndim == 2 and q_values.shape[0] == 1 else q_values

    def get_params(self):
        return {
            'w1': self.w1.copy(),
            'b1': self.b1.copy(),
            'w2': self.w2.copy(),
            'b2': self.b2.copy(),
            'w3': self.w3.copy(),
            'b3': self.b3.copy()
        }

    def set_params(self, params):
        self.w1 = params['w1'].copy()
        self.b1 = params['b1'].copy()
        self.w2 = params['w2'].copy()
        self.b2 = params['b2'].copy()
        self.w3 = params['w3'].copy()
        self.b3 = params['b3'].copy()

    def copy(self):
        clone = DQNNetwork(self.state_size, self.action_size, self.hidden_dim)
        clone.set_params(self.get_params())
        return clone

    def save(self, filepath):
        """Save network weights to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        params = self.get_params()
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)
        print(f"DQN Network saved to {filepath}")

    def load(self, filepath):
        """Load network weights from file"""
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found. Using initialized weights.")
            return
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        self.set_params(params)
        print(f"DQN Network loaded from {filepath}")


class PPONetwork:
    """Simple NumPy-based actor-critic network"""
    def __init__(self, state_size, action_size, hidden_dim=128, seed=None):
        rng = np.random.RandomState(seed)
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_dim = hidden_dim

        self.w1 = rng.randn(state_size, hidden_dim).astype(np.float32) * np.sqrt(2.0 / state_size)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)

        self.actor_w = rng.randn(hidden_dim, action_size).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.actor_b = np.zeros(action_size, dtype=np.float32)

        self.critic_w = rng.randn(hidden_dim, 1).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.critic_b = np.zeros(1, dtype=np.float32)

        self.last_input = None
        self.last_h1 = None

    def forward(self, x):
        x = np.atleast_2d(x).astype(np.float32)
        self.last_input = x
        self.last_h1 = relu(x.dot(self.w1) + self.b1)

        logits = self.last_h1.dot(self.actor_w) + self.actor_b
        values = self.last_h1.dot(self.critic_w) + self.critic_b
        return logits, values

    def predict(self, state):
        return self.forward(state)

    def get_params(self):
        return {
            'w1': self.w1.copy(),
            'b1': self.b1.copy(),
            'actor_w': self.actor_w.copy(),
            'actor_b': self.actor_b.copy(),
            'critic_w': self.critic_w.copy(),
            'critic_b': self.critic_b.copy()
        }

    def set_params(self, params):
        self.w1 = params['w1'].copy()
        self.b1 = params['b1'].copy()
        self.actor_w = params['actor_w'].copy()
        self.actor_b = params['actor_b'].copy()
        self.critic_w = params['critic_w'].copy()
        self.critic_b = params['critic_b'].copy()

    def save(self, filepath):
        """Save network weights to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        params = self.get_params()
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)
        print(f"PPO Network saved to {filepath}")

    def load(self, filepath):
        """Load network weights from file"""
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found. Using initialized weights.")
            return
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        self.set_params(params)
        print(f"PPO Network loaded from {filepath}")
