import numpy as np
import matplotlib.pyplot as plt
import torch

class Custom2DEnv:
    """Custom 2D environment with Gym-like API."""
    def __init__(self, K=0.05):
        self.K = K  
        self.state = np.array([0.0, 0.0])  
        self.trajectory = [self.state.copy()]  

    def reset(self):
        """Resets the environment to the initial state."""
        self.state = np.array([0.0, 0.0])
        self.trajectory = [self.state.copy()] 
        return self.state
    
    def set_state(self, state):
        """Sets the environment state."""
        self.state = state
        self.trajectory.append(state.copy())

    def step(self, action, delta=0.0, sigma=0.2):
        """Performs one step in the environment.
        
        Args:
            action: The action to take (angle in radians).
            delta: Optional noise mean shift (default: 0).
            sigma: Optional noise standard deviation (default: 0.2).
        Returns:
            state: New state (x, y) coordinates.
            action_with_noise: Action after adding noise.
        """
        beta = np.random.normal(delta, sigma)
        
        action_with_noise = action + beta
        new_x = self.state[0] + self.K * np.cos(action_with_noise)
        new_y = self.state[1] + self.K * np.sin(action_with_noise)
        
        self.state = np.array([new_x, new_y])
        self.trajectory.append(self.state.copy())
        
        return self.state, action_with_noise

    def render(self, policy_name=""):
        """Renders the trajectory of the agent in the 2D space."""
        plt.plot(np.array(self.trajectory)[:, 0], np.array(self.trajectory)[:, 1], lw=1, alpha=0.5)
        plt.title(f"Trajectory for {policy_name}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.axis('equal')

    def close(self):
        """Optional clean-up."""
        pass
    
    
class EpsilonPolicy:
    def __init__(self, epsilon, sigma):
        self.epsilon = epsilon
        self.sigma = sigma

    def sample(self, env=None):
        """Sample an action from N(epsilon, sigma^2)."""
        return np.random.normal(self.epsilon, self.sigma)
    
    def log_prob(self, state, action):
        """Compute the log-probability of the action under N(epsilon, sigma^2)."""
        mean = self.epsilon
        variance = self.sigma ** 2
        mean = torch.tensor(mean, dtype=torch.float32).to(action.device)
        variance = torch.tensor(variance, dtype=torch.float32).to(action.device)
        # print(action.requires_grad)
        log_prob = -0.5 * torch.log(2 * torch.pi * variance) - 0.5 * (action - mean) ** 2 / variance
        # print(log_prob.requires_grad)
        return log_prob

class SumPolicy:
    def __init__(self, policy1, policy2, thereshold):
        self.policy1 = policy1
        self.policy2 = policy2
        self.thereshold = thereshold
    
    def sample(self, state):
        return self.policy1.sample(env) if state < self.thereshold else self.policy2.sample(env)
    
    def log_prob(self, state, action):
        log_prob1 = self.policy1.log_prob(state, action)
        log_prob2 = self.policy2.log_prob(state, action)
        #for state[0] less thatn threshold use policy one, otherwise use policy two
        log_prob = torch.where(state[:, 0] < self.thereshold, log_prob1, log_prob2)
        return log_prob
        
        

if __name__ == '__main__':
    test_policy = EpsilonPolicy(epsilon=np.pi/100, sigma=0.2)
    x = torch.randn(2, 3).requires_grad_(True)
    state = x[:, :2]
    action = x[:, 2:]
    log_prob = test_policy.log_prob(state, action)
    log_likelihood = log_prob.sum() / 3
    log_likelihood.backward()
    print(x.grad)