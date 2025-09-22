from enum import Enum
import gym
import numpy as np
from typing import AnyStr, Callable, Dict, List, Tuple
from tqdm import tqdm

from opelab.core.policy import Policy
from opelab.core.reward import RewardEnsembleEstimator
from opelab.core.mlp import MLP

DataType = List[Dict[AnyStr, np.ndarray]]


class Data:
    
    def __init__(self, env:gym.Env, state_enc=None) -> None:
        self.env = env
        if state_enc is None:
            state_enc = lambda s: s
        self.state_enc = state_enc
    
    def make(self, policy:Policy, horizon:int, rollouts:int) -> DataType:
        data = []
        pbar = tqdm(range(rollouts))
        for i in pbar:
            states, actions, rewards = [], [], []
            state = self.env.reset()
            states.append(self.state_enc(state))
            for _ in range(horizon):
                action = policy.sample(state)
                next_state, reward, terminated, *_ = self.env.step(action)
                states.append(self.state_enc(next_state))
                actions.append(action)
                rewards.append(reward)
                if terminated:
                    break
                state = next_state
            states = np.stack(states, axis=0)
            actions = np.stack(actions, axis=0)
            rewards = np.stack(rewards, axis=0)
            data.append({'states': states[:-1, ...],
                         'actions': actions,
                         'rewards': rewards,
                         'next-states': states[1:, ...]})
            
            pbar.set_description(f"Episode {i}, total_reward {np.sum(rewards)}")
        return data
    
    def load_d4rl_dataset(self) -> DataType:
     
        print("Loading D4RL dataset...")
        dataset = self.env.get_dataset()

        states = dataset['observations']
        actions = dataset['actions']
        rewards = dataset['rewards']
        next_states = dataset['next_observations']
        dones = dataset['terminals'] | dataset['timeouts']

        if self.state_enc is not None:
            states = np.array([self.state_enc(s) for s in states])
            next_states = np.array([self.state_enc(s) for s in next_states])

        data = []
        episode_data = {'states': [], 'actions': [], 'rewards': [], 'next-states': []}

        for i in range(len(states)):
            episode_data['states'].append(states[i])
            episode_data['actions'].append(actions[i])
            episode_data['rewards'].append(rewards[i])
            episode_data['next-states'].append(next_states[i])

            if dones[i]:  
                data.append({
                    'states': np.array(episode_data['states']),
                    'actions': np.array(episode_data['actions']),
                    'rewards': np.array(episode_data['rewards']),
                    'next-states': np.array(episode_data['next-states'])
                })
                episode_data = {'states': [], 'actions': [], 'rewards': [], 'next-states': []}

        print(f"Loaded {len(data)} episodes from D4RL dataset.")
        return data
    
    def load_dataset(self, path: AnyStr) -> DataType:
        print(f"Loading dataset from {path}...")
        observations = np.load(f"{path}/observations.npy")
        actions = np.load(f"{path}/actions.npy")
        rewards = np.load(f"{path}/rewards.npy")
        terminals = np.load(f"{path}/terminals.npy")
        
        if self.state_enc is not None:
            observations = np.array([self.state_enc(s) for s in observations])
        
        data = []
        episode_data = {'states': [], 'actions': [], 'rewards': [], 'next-states': []}
        terminal_flag = False
        for i in range(len(observations)):
            
            #This is to ensure we dont use the last state where next_state is not available
            if terminal_flag:
                terminal_flag = False
                continue
            
            episode_data['states'].append(observations[i])
            episode_data['actions'].append(actions[i])
            episode_data['rewards'].append(rewards[i])
            episode_data['next-states'].append(observations[i+1] if i+1 < len(observations) else observations[i])

            if terminals[i+1]:
                data.append({
                    'states': np.array(episode_data['states'], dtype=np.float32),
                    'actions': np.array(episode_data['actions'], dtype=np.float32),
                    'rewards': np.array(episode_data['rewards'], dtype=np.float32),
                    'next-states': np.array(episode_data['next-states'], dtype=np.float32)
                })
                episode_data = {'states': [], 'actions': [], 'rewards': [], 'next-states': []}
                terminal_flag = True
                
        print(f"Loaded {len(data)} episodes from dataset.")
        return data
                        
    def train_reward_estimator(self, data: DataType):
        print('Training the reward Estimator')
        rewards_x = []
        rewards_y = []
        for tao in data:
            states = tao['states']
            actions = tao['actions']
            rewards = tao['rewards']
            for i in range(len(states)):
                rewards_x.append((np.concatenate([states[i], actions[i]]).tolist()))
                rewards_y.append(rewards[i])
        rewards_x = np.array(rewards_x)
        rewards_y = np.array(rewards_y)
        mlp = MLP([64, 64, 1])
        est = RewardEnsembleEstimator(mlp)
        est.fit(rewards_x, rewards_y, 1000, [42])
        return est


Normalization = Enum('Normalization', ['NORM', 'STD', 'NONE'])


def to_numpy(data: DataType, target: Policy, behavior: Policy, 
             normalization: Normalization=Normalization.STD, 
             clip_ratio_min: float=0.0, clip_ratio_max: float=np.inf, return_terminals=False) -> Tuple:
    states, actions, next_states, rewards, target_prob, behavior_prob, terminals = [], [], [], [], [], [], []
    n = 0
    for tau in tqdm(data):
        states.append(tau['states'])
        actions.append(tau['actions'])
        next_states.append(tau['next-states'])
        rewards.append(tau['rewards'])
        n += len(tau['rewards'])
        target_prob.extend([target.prob(s, a) for s, a in zip(tau['states'], tau['actions'])])
        behavior_prob.extend([behavior.prob(s, a) for s, a in zip(tau['states'], tau['actions'])])
    
        trmnl = [0 for i in range(len(tau['states']))]
        trmnl[-1] = 1 
        terminals.extend(np.array(trmnl))
    
    terminals = np.concatenate([terminals], axis=0).reshape((n,1))
    states = np.concatenate(states, axis=0).reshape((n, -1))
    actions = np.concatenate(actions, axis=0).reshape((n, -1))
    next_states = np.concatenate(next_states, axis=0).reshape((n, -1))
    states_un, next_states_un = states, next_states
    if normalization == Normalization.STD:
        s_mean = np.mean(states, axis=0, keepdims=True)
        s_std = np.std(states, axis=0, keepdims=True)
        states = (states - s_mean) / s_std
        next_states = (next_states - s_mean) / s_std
    elif normalization == Normalization.NORM:
        s_min = np.min(states, axis=0, keepdims=True)
        s_max = np.max(states, axis=0, keepdims=True)
        states = (states - s_min) / (s_max - s_min)
        next_states = (next_states - s_min) / (s_max - s_min)
    rewards = np.concatenate(rewards, axis=0).reshape((n, 1))
    target_prob = np.asarray(target_prob).reshape((n, 1))
    behavior_prob = np.asarray(behavior_prob).reshape((n, 1))
    if target_prob.any() == None or behavior_prob.any() == None:
        policy_ratio = None
    else:
        policy_ratio = np.clip(target_prob / behavior_prob, clip_ratio_min, clip_ratio_max)
    if return_terminals:
            return states, states_un, actions, next_states, next_states_un, rewards, policy_ratio, terminals
    return states, states_un, actions, next_states, next_states_un, rewards, policy_ratio
