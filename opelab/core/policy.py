import numpy as np
import math
from typing import Any, Sequence
import torch
from torch.distributions import MultivariateNormal, Normal
import torch.nn as nn
import torch.nn.functional as F
import gym
import d4rl
import pickle
import os
# from cleandiffuser.nn_condition import PearceObsCondition
# from cleandiffuser.nn_diffusion import PearceMlp
# from cleandiffuser.diffusion.diffusionsde import DiscreteDiffusionSDE


from opelab.core.baselines.td3.TD3 import Actor

class Policy:
    
    def sample(self, state:Any, deterministic:bool=False, vectorize=False) -> Any:
        raise NotImplementedError
    
    def prob(self, state:Any, action:Any) -> float:
        raise NotImplementedError
    
    def visualize(self, env: gym.Env, horizon:int, episodes:int=1) -> None:
        for _ in range(episodes):
            state, _ = env.reset()
            for _ in range(horizon):
                env.render()
                action = self.sample(state)
                state, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break            


class MixturePolicy(Policy):
    
    def __init__(self, base_policies:Sequence[Policy], weights:np.ndarray) -> None:
        self.base_policies = base_policies
        self.weights = np.reshape(weights, newshape=(-1,))
        assert len(self.base_policies) == len(self.weights)
        assert np.allclose(np.sum(self.weights), 1)
        
    def sample(self, state:Any, deterministic:bool=False) -> Any:
        index = np.random.choice(len(self.base_policies), p=self.weights)
        policy = self.base_policies[index]
        return policy.sample(state, deterministic)
    
    def prob(self, state:Any, action:Any) -> float:
        probs = [pi.prob(state, action) for pi in self.base_policies]
        probs = np.asarray(probs)
        return np.sum(self.weights * probs, axis=0)

    def log_prob(self, state: Any, action: Any) -> float:
        """
        More Numerical Stable Version
        """

        log_probs = []
        for i in range(len(self.base_policies)):
            log_prob_i = self.base_policies[i].log_prob(state, action) + math.log(self.weights[i].item() + 1e-40)
            log_probs.append(log_prob_i)
        log_probs = torch.stack(log_probs).squeeze()
        log_prob_final = torch.logsumexp(log_probs, 0)
        return log_prob_final     


    def gaussian_prob(self, state:Any, action:Any) -> float:
        probs = [pi.gaussian_prob(state, action) for pi in self.base_policies]
        probs = np.asarray(probs)
        final_prob = np.sum(self.weights.reshape(-1, 1) * probs.reshape(len(self.weights), -1), axis=0)
        return final_prob
    
    def to(self, device):
        for pi in self.base_policies:
            pi.to(device)
        return self

    # def gaussian_log_prob(self, state:Any, action:Any) -> float: #I added this for now but it shouldnt be like this
    #     log_probs = [pi.gaussian_log_prob(state, action) for pi in self.base_policies]
    #     print('log probs',log_probs)
    #     log_probs = torch.exp(torch.stack(log_probs))
    #     print('probs', log_probs)
    #     return torch.log(torch.matmul(torch.tensor(self.weights, device=log_probs.device, dtype=torch.float32), log_probs.squeeze()))

    def gaussian_log_prob(self, state: Any, action: Any) -> float:
        """
        More Numerical Stable Version
        """

        log_probs = []
        for i in range(len(self.base_policies)):
            log_prob_i = self.base_policies[i].gaussian_log_prob(state, action) + math.log(self.weights[i].item() + 1e-40)
            log_probs.append(log_prob_i)
        log_probs = torch.stack(log_probs).squeeze()
        log_prob_final = torch.logsumexp(log_probs, 0)
        return log_prob_final       

class Boltzmann(Policy):
    
    def __init__(self, critic, temp:float) -> None:
        self.critic = critic
        self.temp = temp
    
    def _boltzmann(self, x):
        t = x / self.temp
        p = np.exp(t - 0.5 * (np.amax(t) + np.amin(t)))
        return p / np.sum(p)
    
    def sample(self, state:Any, deterministic:bool=False) -> Any:
        if deterministic:
            return np.argmax(self.critic.values(state))
        else:
            p = self._boltzmann(self.critic.values(state))
            return np.random.choice(p.shape[0], p=p)
    
    def prob(self, state:Any, action:Any) -> float:
        return self._boltzmann(self.critic.values(state))[action]


class EpsilonGreedy(Policy):
    
    def __init__(self, critic, eps:float) -> None:
        self.critic = critic
        self.eps = eps
    
    def sample(self, state:Any, deterministic:bool=False) -> Any:
        v = self.critic.values(state)
        if deterministic or np.random.rand() <= 1.0 - self.eps:
            return np.argmax(v)
        else:
            return np.random.randint(0, v.size)
    
    def prob(self, state:Any, action:Any) -> float:
        v = self.critic.values(state)
        if action == np.argmax(v):
            return 1.0 - self.eps + self.eps / v.size
        else:
            return self.eps / v.size

    
class Uniform(Policy):
    
    def __init__(self, num_actions) -> None:
        self.num_actions = num_actions
    
    def sample(self, state:Any, deterministic:bool=False) -> Any:
        return np.random.randint(0, self.num_actions)
    
    def prob(self, state:Any, action:Any) -> float:
        return np.prod(1.0 / self.num_actions)


class UniformContinuous(Policy):
    
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def sample(self, state, deterministic=False):
        return np.random.uniform(low=self.lower, high=self.upper)

    def prob(self, state, action):
        return np.prod(1. / (self.upper - self.lower))


class Greedy(Policy):
    
    def __init__(self, critic) -> None:
        self.critic = critic
    
    def sample(self, state:Any, deterministic:bool=False) -> Any:
        return np.argmax(self.critic.values(state))
    
    def prob(self, state:Any, action:Any) -> float:
        values = self.critic.values(state).reshape((-1,))
        if values[action] >= np.amax(values):
            return 1.0
        else:
            return 0.0
        

class TD3Policy(Policy):
    def __init__(self, model_path, std: float = 1.0, action_bound = 1.0, device='cpu') -> None:
        self.model_path = model_path
        self.std = std
        self.device = device
        self.action_bound = action_bound
        
        # Load weights
        with open(model_path, 'rb') as f:
            weights = pickle.load(f)
        
        # Create tensors from weights
        self.l1_w = torch.tensor(weights['l1/weight'], dtype=torch.float32)
        self.l1_b = torch.tensor(weights['l1/bias'], dtype=torch.float32)
        self.l2_w = torch.tensor(weights['l2/weight'], dtype=torch.float32)
        self.l2_b = torch.tensor(weights['l2/bias'], dtype=torch.float32)
        self.l3_w = torch.tensor(weights['l3/weight'], dtype=torch.float32)
        self.l3_b = torch.tensor(weights['l3/bias'], dtype=torch.float32)
        self.max_action = weights['max_action']
        
        # Move to device
        self.to(device)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(torch.matmul(state, self.l1_w.t()) + self.l1_b)
        x = F.relu(torch.matmul(x, self.l2_w.t()) + self.l2_b)
        x = self.max_action * torch.tanh(torch.matmul(x, self.l3_w.t()) + self.l3_b)
        return x

    def sample(self, state: Any, deterministic: bool = False) -> Any:
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).to(self.device).squeeze()
        if state.ndim == 1:
            state = state.unsqueeze(0)
            
        action = self.forward(state)
        if not deterministic:
            action += torch.normal(mean=0, std=self.std, size=action.size()).to(self.device)
        return action.view(-1).cpu().detach().numpy()

    def prob(self, state: Any, action: Any) -> float:
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32).to(self.device)
            
        mean = self.forward(state)
        dist = Normal(mean, self.std)
        return dist.log_prob(action).exp().cpu().item()
    
    def vectorized_prob(self, state: Any, action: Any) -> float:
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32).to(self.device)
            
        mean = self.forward(state)
        dist = Normal(mean, self.std)
        return dist.log_prob(action).exp().cpu()

    def log_prob(self, state: Any, action: Any) -> float:
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32).to(self.device)
            
        mean = self.forward(state)
        dist = Normal(mean, self.std)
        return dist.log_prob(action)

    def log_prob_extended(self, state: Any, action: Any) -> float:
            
        state = state.to(dtype=torch.float32)
        action = action.to(dtype=torch.float32)
                
        mean = self.forward(state)
        dist = Normal(mean, self.std)
        return dist.log_prob(action)
    
    def grad_log_prob_extended_pgd(self, state: np.ndarray or torch.Tensor, action: np.ndarray or torch.Tensor) -> torch.Tensor:
        """
        Computes the gradient of the log probability of the action w.r.t. the action.
        """
        state = np.asarray(state)
        action = np.asarray(action)
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).to('cuda:0')
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32).to('cuda:0').requires_grad_()
        
        log_p = self.log_prob_extended(state, action * self.action_bound).sum()
        log_p.backward()
        gradient = action.grad.detach().cpu().numpy()
        return gradient

    
    def to(self, device):
        self.l1_w = self.l1_w.to(device)
        self.l1_b = self.l1_b.to(device)
        self.l2_w = self.l2_w.to(device)
        self.l2_b = self.l2_b.to(device)
        self.l3_w = self.l3_w.to(device)
        self.l3_b = self.l3_b.to(device)
        self.device = device
        return self
    

class D4RLPolicy:
    """D4RL policy."""

    def __init__(self, env_name):
        self.env_name = env_name
        
        # Parse the environment name to detect medium-expert type
        parts = env_name.split('-')
        if len(parts) >= 3 and parts[-1].startswith('v'):
            base_env = parts[0]
            type_ = "-".join(parts[1:-1])    
            version = parts[-1]              
        else:
            raise ValueError(f"Invalid environment name format: {env_name}")
        
        print(type_)

        if type_ == 'medium-expert':
            medium_env = f"{base_env}-medium-{version}"
            expert_env = f"{base_env}-expert-{version}"
            self.medium_policy = D4RLPolicy(medium_env)
            self.expert_policy = D4RLPolicy(expert_env)
            self.is_mixture = True
        else:
            self.is_mixture = False
            env = gym.make(env_name)
            dataset = env.get_dataset()
            print(dataset.keys())
            
            weights = self.load_policy_from_metadata(dataset)  
            
            self.fc0_w = torch.tensor(weights['fc0/weight']).t()
            self.fc0_b = torch.tensor(weights['fc0/bias'])
            self.fc1_w = torch.tensor(weights['fc1/weight']).t()
            self.fc1_b = torch.tensor(weights['fc1/bias'])
            self.fclast_w = torch.tensor(weights['last_fc/weight']).t()
            self.fclast_b = torch.tensor(weights['last_fc/bias'])
            self.fclast_w_logstd = torch.tensor(weights['last_fc_log_std/weight']).t()
            self.fclast_b_logstd = torch.tensor(weights['last_fc_log_std/bias'])
            relu = lambda x: torch.maximum(x, torch.tensor(0.0))
            self.nonlinearity = torch.tanh if weights['nonlinearity'] == 'tanh' else relu

            identity = lambda x: x
            self.output_transformation = torch.tanh if weights['output_distribution'] == 'tanh_gaussian' else identity
                
            self.check_metadata_consistency(dataset)

    def to(self, device):
        if self.is_mixture:
            self.medium_policy.to(device)
            self.expert_policy.to(device)
        else:
            self.fc0_w = self.fc0_w.to(device)
            self.fc0_b = self.fc0_b.to(device)
            self.fc1_w = self.fc1_w.to(device)
            self.fc1_b = self.fc1_b.to(device)
            self.fclast_w = self.fclast_w.to(device)
            self.fclast_b = self.fclast_b.to(device)
            self.fclast_w_logstd = self.fclast_w_logstd.to(device)
            self.fclast_b_logstd = self.fclast_b_logstd.to(device)
        return self

    def sample(self, state, deterministic=False):
        return self.act(state, deterministic)[0].cpu().detach().numpy()
    
    def vectorized_prob(self, state, action):
        if self.is_mixture:
            prob1 = self.medium_policy.vectorized_prob(state, action)
            prob2 = self.expert_policy.vectorized_prob(state, action)
            return 0.5 * prob1 + 0.5 * prob2
        else:
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32, device=self.fc0_w.device)
                action = torch.tensor(action, dtype=torch.float32, device=self.fc0_w.device)
            return torch.exp(self.log_prob(state, action).sum(dim=-1))
    
    def prob(self, state, action):
        if self.is_mixture:
            prob1 = self.medium_policy.prob(state, action)
            prob2 = self.expert_policy.prob(state, action)
            return 0.5 * prob1 + 0.5 * prob2
        else:
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32, device=self.fc0_w.device)
                action = torch.tensor(action, dtype=torch.float32, device=self.fc0_w.device)
            return torch.exp(self.log_prob(state, action).sum()).item()

    def act(self, state, deterministic=False):
        if self.is_mixture:
            # 50% chance to use medium or expert policy
            if np.random.rand() < 0.5:
                return self.medium_policy.act(state, deterministic)
            else:
                return self.expert_policy.act(state, deterministic)
        else:
            state = torch.tensor(state, dtype=torch.float32, device=self.fc0_w.device)
            
            x = torch.matmul(state, self.fc0_w) + self.fc0_b
            x = self.nonlinearity(x)
            x = torch.matmul(x, self.fc1_w) + self.fc1_b
            x = self.nonlinearity(x)
            mean = torch.matmul(x, self.fclast_w) + self.fclast_b
            logstd = torch.matmul(x, self.fclast_w_logstd) + self.fclast_b_logstd
            std = torch.exp(logstd)
            
            if deterministic:
                action = mean
            else:
                noise = torch.randn_like(mean)
                action = mean + std * noise
            
            action = self.output_transformation(action)
            return action, mean

    def log_prob(self, state, action):
        if self.is_mixture:
            log_p1 = self.medium_policy.log_prob(state, action)
            log_p2 = self.expert_policy.log_prob(state, action)
            # Mixture log probability: log(0.5 * p1 + 0.5 * p2)
            log_prob = torch.log(0.5) + torch.logsumexp(torch.stack([log_p1, log_p2], dim=0), dim=0)
            return log_prob
        else:
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32, device=self.fc0_w.device)
                action = torch.tensor(action, dtype=torch.float32, device=self.fc0_w.device)
            x = torch.matmul(state, self.fc0_w) + self.fc0_b
            x = self.nonlinearity(x)
            x = torch.matmul(x, self.fc1_w) + self.fc1_b
            x = self.nonlinearity(x)
            mean = torch.matmul(x, self.fclast_w) + self.fclast_b
            logstd = torch.matmul(x, self.fclast_w_logstd) + self.fclast_b_logstd

            logstd = torch.clamp(logstd, min=-20, max=2)        
            std = torch.exp(logstd)
            
            u = torch.atanh(action.clamp(-0.999999, 0.999999))
            log_prob = -0.5 * (((u - mean) / (std + 1e-10)) ** 2 + 2 * logstd + np.log(2 * np.pi))
            
            if self.output_transformation == torch.tanh:
                log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            
            return log_prob
    
    def log_prob_extended(self, state, action):
        if self.is_mixture:
            log_p1 = self.medium_policy.log_prob_extended(state, action)
            log_p2 = self.expert_policy.log_prob_extended(state, action)
            log_prob = torch.log(0.5) + torch.logsumexp(torch.stack([log_p1, log_p2], dim=0), dim=0)
            return log_prob
        else:
            x = torch.matmul(state, self.fc0_w) + self.fc0_b
            x = self.nonlinearity(x)
            x = torch.matmul(x, self.fc1_w) + self.fc1_b
            x = self.nonlinearity(x)
            mean = torch.matmul(x, self.fclast_w) + self.fclast_b
            std = torch.tensor(1, device=mean.device)
            
            mean_action = torch.tanh(mean)
            mean_action = torch.clamp(mean_action, min=-10, max=10)
            
            log_prob = -0.5 * (((action - mean_action) / std) ** 2 + 2 * torch.log(std) + np.log(2 * np.pi))
            return log_prob
    
    def grad_log_prob_extended(self, state, action):
        if self.is_mixture:
            grad_log_p1 = self.medium_policy.grad_log_prob_extended(state, action)
            grad_log_p2 = self.expert_policy.grad_log_prob_extended(state, action)
            return 0.5 * grad_log_p1 + 0.5 * grad_log_p2
        else:
            x = torch.matmul(state, self.fc0_w) + self.fc0_b
            x = self.nonlinearity(x)
            x = torch.matmul(x, self.fc1_w) + self.fc1_b
            x = self.nonlinearity(x)
            mean = torch.matmul(x, self.fclast_w) + self.fclast_b
            logstd = torch.matmul(x, self.fclast_w_logstd) + self.fclast_b_logstd
            std = torch.exp(logstd)
            
            std = torch.max(std / torch.cosh(mean) ** 2, torch.tensor(1, device=mean.device))
            mean_action = torch.tanh(mean)
            
            grad_log_prob = -((action - mean_action) / std ** 2)
            return grad_log_prob
    
    def gaussian_log_prob(self, state, action):
        if self.is_mixture:
            log_p1 = self.medium_policy.gaussian_log_prob(state, action)
            log_p2 = self.expert_policy.gaussian_log_prob(state, action)
            log_prob = torch.log(0.5) + torch.logsumexp(torch.stack([log_p1, log_p2], dim=0), dim=0)
            return log_prob
        else:
            x = torch.matmul(state, self.fc0_w) + self.fc0_b
            x = self.nonlinearity(x)
            x = torch.matmul(x, self.fc1_w) + self.fc1_b
            x = self.nonlinearity(x)
            mean = torch.matmul(x, self.fclast_w) + self.fclast_b
            logstd = torch.matmul(x, self.fclast_w_logstd) + self.fclast_b_logstd
            logstd = torch.clamp(logstd, min=-20, max=2)
            std = torch.exp(logstd) + 1e-6
            
            log_prob = -0.5 * (((action - mean) / std) ** 2 + 2 * logstd + np.log(2 * np.pi))
            return log_prob

    def load_policy_from_metadata(self, dataset):
        weights = {
            'fc0/weight': dataset['metadata/policy/fc0/weight'],
            'fc0/bias': dataset['metadata/policy/fc0/bias'],
            'fc1/weight': dataset['metadata/policy/fc1/weight'],
            'fc1/bias': dataset['metadata/policy/fc1/bias'],
            'last_fc/weight': dataset['metadata/policy/last_fc/weight'],
            'last_fc/bias': dataset['metadata/policy/last_fc/bias'],
            'last_fc_log_std/weight': dataset['metadata/policy/last_fc_log_std/weight'],
            'last_fc_log_std/bias': dataset['metadata/policy/last_fc_log_std/bias'],
            'nonlinearity': dataset['metadata/policy/nonlinearity'].decode('utf-8'),
            'output_distribution': dataset['metadata/policy/output_distribution'].decode('utf-8')
        }
        return weights

    def check_metadata_consistency(self, dataset):
        terminals = dataset['terminals']
        timeouts = dataset['timeouts']
        episode_indices = np.where(terminals | timeouts)[0]
        if len(episode_indices) == 0:
            print("No complete episodes found in the dataset.")
            return

        random_episode_idx = np.random.choice(episode_indices)
        
        start_idx = random_episode_idx
        while start_idx > 0 and not terminals[start_idx - 1] and not timeouts[start_idx - 1]:
            start_idx -= 1
        
        states = dataset['observations'][start_idx:random_episode_idx + 1]
        actions = dataset['actions'][start_idx:random_episode_idx + 1]
        
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        
        total_log_prob = 0.0
        for state, action in zip(states, actions):
            log_prob = self.log_prob(state, action)
            total_log_prob += log_prob.sum().item()
        
        print(f"Total log probability for the episode: {total_log_prob}")
        
    def get_info(self, state):
        if self.is_mixture:
            print("Medium policy info:")
            self.medium_policy.get_info(state)
            print("Expert policy info:")
            self.expert_policy.get_info(state)
        else:
            x = torch.matmul(state, self.fc0_w) + self.fc0_b
            x = self.nonlinearity(x)
            x = torch.matmul(x, self.fc1_w) + self.fc1_b
            x = self.nonlinearity(x)
            mean = torch.matmul(x, self.fclast_w) + self.fclast_b
            logstd = torch.matmul(x, self.fclast_w_logstd) + self.fclast_b_logstd
            std = torch.exp(logstd)
            
            print(f'Mean: {mean}, Std: {std}')
            action = torch.tanh(mean)
            print(f'Action: {action}')
            log_prob = self.log_prob(state, action)
            

class D4RLSACPolicy(Policy):
    """D4RL policy."""

    def __init__(self, policy_file: str) -> None:
        with open(policy_file, 'rb') as f:
            weights = pickle.load(f)

        # Load weights
        self.fc0_w = torch.tensor(weights['fc0/weight'], dtype=torch.float32)
        self.fc0_b = torch.tensor(weights['fc0/bias'], dtype=torch.float32)
        self.fc1_w = torch.tensor(weights['fc1/weight'], dtype=torch.float32)
        self.fc1_b = torch.tensor(weights['fc1/bias'], dtype=torch.float32)
        self.fclast_w = torch.tensor(weights['last_fc/weight'], dtype=torch.float32)
        self.fclast_b = torch.tensor(weights['last_fc/bias'], dtype=torch.float32)
        self.fclast_w_logstd = torch.tensor(weights['last_fc_log_std/weight'], dtype=torch.float32)
        self.fclast_b_logstd = torch.tensor(weights['last_fc_log_std/bias'], dtype=torch.float32)

        # Nonlinearity
        if weights['nonlinearity'] == 'tanh':
            self.nonlinearity = torch.tanh
        else:
            self.nonlinearity = nn.ReLU()

        # Output distribution	
        if weights['output_distribution'] == 'tanh_gaussian':
            self.output_transformation = torch.tanh
        else:
            self.output_transformation = lambda x: x

        # If you want to set device usage, you can do so here
        # e.g., self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, state: torch.Tensor):
        """
        A small forward pass that returns mean and logstd before the final output transformation.

        Args:
            state (torch.Tensor): The input state (or batch of states).
        Returns:
            (mean, logstd): Tensors of shape matching the action dimension.
        """
        x = torch.matmul(state, self.fc0_w.t()) + self.fc0_b
        x = self.nonlinearity(x)
        x = torch.matmul(x, self.fc1_w.t()) + self.fc1_b
        x = self.nonlinearity(x)

        mean = torch.matmul(x, self.fclast_w.t()) + self.fclast_b
        logstd = torch.matmul(x, self.fclast_w_logstd.t()) + self.fclast_b_logstd
        return mean, logstd

    def act(self, state: torch.Tensor, noise: torch.Tensor):
        """
        Original method from your snippet. Combines mean + noise*std, then does a tanh transform if configured.

        Args:
            state (torch.Tensor): State tensor.
            noise (torch.Tensor): Random noise (same shape as mean).
        Returns:
            (action, mean): The transformed action and the raw mean.
        """
        mean, logstd = self.forward(state)
        std = torch.exp(logstd)
        raw_action = mean + std * noise
        action = self.output_transformation(raw_action)
        return action, mean

    def sample(self, state: np.ndarray or torch.Tensor, deterministic: bool = False):
        """
        Samples an action from the policy at the given state.

        Args:
            state: A NumPy array or torch.Tensor (shape [state_dim]) or a batch of states.
            deterministic: If True, use the mean (no added noise). If False, sample with noise.
        Returns:
            (torch.Tensor or np.ndarray): Sampled action(s).
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.fc0_w.device)

        mean, logstd = self.forward(state)
        std = torch.exp(logstd)

        if deterministic:
            noise = torch.zeros_like(mean)
        else:
            noise = torch.randn_like(mean)

        raw_action = mean + std * noise
        action = self.output_transformation(raw_action)
        return action.cpu().detach().numpy()

    def log_prob(self, state: np.ndarray or torch.Tensor, action: np.ndarray or torch.Tensor) -> torch.Tensor:
        """
        Computes log p(a|s). For a tanh-Gaussian distribution, we do the usual correction.

        Args:
            state: [N, state_dim] or [state_dim].
            action: [N, action_dim] or [action_dim].
        Returns:
            log_prob: A tensor of shape [N] or scalar if state/action is a single example.
        """

        mean, logstd = self.forward(state)
        # Clamp logstd to avoid numerical blow-ups
        logstd = torch.clamp(logstd, min=-20, max=2)
        std = torch.exp(logstd)

        if self.output_transformation == torch.tanh:
            eps = 1e-6
            action_clamped = torch.clamp(action, -1 + eps, 1 - eps)
            u = 0.5 * torch.log((1 + action_clamped) / (1 - action_clamped))

            gaussian_log_prob = -0.5 * (((u - mean) / std) ** 2 + 2 * logstd + np.log(2.0 * np.pi))
            # Tanh correction: log( d(u)/d(a) ) = log(1/(1 - a^2)) => - log(1 - a^2)
            correction = -torch.log(1 - action_clamped * action_clamped + eps)
            log_p = gaussian_log_prob.sum(dim=-1) + correction.sum(dim=-1)
        else:
            gaussian_log_prob = -0.5 * (((action - mean) / std) ** 2 + 2 * logstd + np.log(2.0 * np.pi))
            log_p = gaussian_log_prob.sum(dim=-1)

        return log_p

    def log_prob_extended(self, state: np.ndarray or torch.Tensor, action: np.ndarray or torch.Tensor) -> torch.Tensor:
        """
        Example 'extended' version of log_prob that you might customize for additional transformations.
        In this sample, we'll do the same as log_prob but clamp the standard deviation differently
        or do a different computation. Modify as needed.
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32)

        mean, logstd = self.forward(state)
        # Potentially do some custom clamp or other modifications:
        logstd = torch.clamp(logstd, min=-20, max=2)
        std = torch.exp(logstd)

        #create around the mean with 1 std
        mean_action = torch.tanh(mean)
        std = torch.tensor(1, device=mean.device)
        log_prob = -0.5 * (((action - mean_action) / std) ** 2 + 2 * torch.log(std) + np.log(2 * np.pi))
        return log_prob
    
    def gaussian_log_prob(self, state: np.ndarray or torch.Tensor, action: np.ndarray or torch.Tensor) -> torch.Tensor:
        """
        Computes the log probability of a Gaussian distribution.
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32)

        mean, logstd = self.forward(state)
        logstd = torch.clamp(logstd, min=-20, max=2)
        std = torch.exp(logstd)
        gaussian_log_prob = -0.5 * (((action - mean) / std) ** 2 + 2 * logstd + np.log(2.0 * np.pi))
        return gaussian_log_prob
    
    def load_policy(self, policy_file: str) -> None:
        with open(policy_file, 'rb') as f:
            weights = pickle.load(f)
                
        
    def prob(self, state: np.ndarray or torch.Tensor, action: np.ndarray or torch.Tensor) -> float:
        """
        Probability of taking `action` in state `state`. 
        Typically returns a floating scalar if state/action is single, 
        or a vector if batched.
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.fc0_w.device)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32, device=self.fc0_w.device)
            
        # exponentiate the log_prob
        log_p = self.log_prob(state, action)
        # Return as float if single, else a torch.Tensor
        if log_p.ndim == 0:
            return torch.exp(log_p).item()
        else:
            return torch.exp(log_p)
        
    def grad_log_prob(self, state: np.ndarray or torch.Tensor, action: np.ndarray or torch.Tensor) -> torch.Tensor:
        """
        Computes the gradient of the log probability of the action w.r.t. the action.
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).to(self.fc0_w.device).requires_grad_()
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32).to(self.fc0_w.device).requires_grad_()

        log_p = self.log_prob(state, action).sum()
        log_p.backward()
        return action.grad
    
    def grad_log_prob_extended(self, state: np.ndarray or torch.Tensor, action: np.ndarray or torch.Tensor) -> torch.Tensor:
        """
        Computes the gradient of the log probability of the action w.r.t. the action.
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).to(self.fc0_w.device).requires_grad_()
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32).to(self.fc0_w.device).requires_grad_()

        log_p = self.log_prob_extended(state, action).sum()
        log_p.backward()
        return action.grad

    def grad_log_prob_extended_pgd(self, state: np.ndarray or torch.Tensor, action: np.ndarray or torch.Tensor) -> torch.Tensor:
        """
        Computes the gradient of the log probability of the action w.r.t. the action.
        """
        state = np.asarray(state)
        action = np.asarray(action)
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).to('cuda:0')
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32).to('cuda:0').requires_grad_()

        log_p = self.log_prob_extended(state, action).sum()
        log_p.backward()
        gradient = action.grad.detach().cpu().numpy()
        return gradient

    def to(self, device: torch.device):
        """
        Optionally move all parameters to a specified device (cpu/cuda).
        """
        self.fc0_w = self.fc0_w.to(device)
        self.fc0_b = self.fc0_b.to(device)
        self.fc1_w = self.fc1_w.to(device)
        self.fc1_b = self.fc1_b.to(device)
        self.fclast_w = self.fclast_w.to(device)
        self.fclast_b = self.fclast_b.to(device)
        self.fclast_w_logstd = self.fclast_w_logstd.to(device)
        self.fclast_b_logstd = self.fclast_b_logstd.to(device)
        return self
    
class DiffusionPolicy(Policy):

    def __init__(self, obs_dim: int, act_dim: int, policy_path: str = None, device=None):
        print('using diffusion policy on device:', device)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.nn_diffusion = PearceMlp(act_dim, To=1, emb_dim=64, hidden_dim=256, timestep_emb_type="positional")
        self.nn_condition = PearceObsCondition(obs_dim, emb_dim=64, flatten=True, dropout=0.0)

        self.actor = DiscreteDiffusionSDE(
            self.nn_diffusion, self.nn_condition,
            predict_noise=True, optim_params={"lr": 3e-4},
            x_max=+1. * torch.ones((1, act_dim)),
            x_min=-1. * torch.ones((1, act_dim)),
            diffusion_steps=32, 
            ema_rate=0.9999, device=self.device
        )

        self.load(policy_path)

        self.to(self.device)

    def sample(self, state: np.ndarray or torch.Tensor, deterministic: bool = False):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            if state.ndim == 1:
                state = state.unsqueeze(0)
        
        prior = torch.zeros((state.shape[0], self.actor.x_max.shape[-1]), device=self.device)

        with torch.no_grad():
            action, _ = self.actor.sample(
                prior, solver="ddpm", n_samples=state.shape[0], sample_steps=32,
                temperature=0.0 if deterministic else 0.5,  # No noise if deterministic
                w_cfg=1.0, condition_cfg=state
            )

        return action.cpu().numpy().squeeze()
    
    def sample_tensor(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
  
        if state.ndim == 1:
            state = state.unsqueeze(0)
        # create a zero‐prior on the same device
        prior = torch.zeros((state.shape[0], self.actor.x_max.shape[-1]),
                             device=state.device)
        
        with torch.no_grad():
            action, _ = self.actor.sample(
                prior,
                solver="ddpm",
                n_samples=state.shape[0],
                sample_steps=self.actor.diffusion_steps,
                temperature=0.0 if deterministic else 0.5,
                w_cfg=1.0,
                condition_cfg=state
            )
        # remove any extra dims
        return action.squeeze(0) if action.ndim > state.ndim else action

    
    def prob(self, state: np.ndarray or torch.Tensor, action: np.ndarray or torch.Tensor) -> float:
        return None
    
    def log_prob(self, state: np.ndarray or torch.Tensor, action: np.ndarray or torch.Tensor) -> float:
        return None

    def load(self, filepath: str):
        self.actor.load(filepath)
        self.actor.eval()
        print(f"Policy loaded from {filepath}")

    def to(self, device: torch.device):
        self.device = device
        return self
    
    def grad_log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        cond_net = self.actor.model['condition']
        diffusion_net = self.actor.model['diffusion']        
        score_fn = diffusion_net(action, torch.zeros(action.shape[0], device=action.device) + 1 ,cond_net(state)) 
        # print(score_fn)
        return -score_fn / self.actor.sigma[1]

        
        