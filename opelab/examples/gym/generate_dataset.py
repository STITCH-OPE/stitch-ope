import os
import numpy as np
import torch
import gym
import argparse
from stable_baselines3 import PPO
from tqdm import tqdm
import sys
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../../..'))

from opelab.core.policy import TD3Policy
from opelab.core.task import ContinuousAcrobotEnv

CHOICES = ['PPO', 'TD3', 'SAC']
AGENT = 'TD3'

def generate_trajectories_d4rl(env, policy, num_episodes, T):
    trajectories = []
    
    for episode in tqdm(range(num_episodes)):
        obs = env.reset()
        trajectory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'terminals': [],          
        }
        for t in range(T):
            action = policy.sample(obs.flatten(), deterministic=False)
            next_obs, reward, done, _ = env.step(action)
            done = done or t == T - 1
            
            trajectory['observations'].append(obs.flatten())
            trajectory['actions'].append(action.flatten())
            trajectory['rewards'].append(reward)
            trajectory['terminals'].append(done)
        
            obs = next_obs
            if done:
                break
        
        # Convert lists to arrays
        for key in trajectory:
            trajectory[key] = np.array(trajectory[key])
        trajectories.append(trajectory)
    
    return trajectories

def evaluate_trajectories(trajectories):
    total_reward = 0
    total_length = 0
    for traj in trajectories:
        total_reward += np.sum(traj['rewards'])
        total_length += len(traj['rewards'])
    
    print(f'Average reward: {total_reward / len(trajectories)}')
    print(f'Average length: {total_length / len(trajectories)}')
    
    concat_states = np.concatenate([traj['observations'] for traj in trajectories])
    concat_actions = np.concatenate([traj['actions'] for traj in trajectories])
    print(concat_states.shape)
    return np.mean(concat_states, axis=0), np.std(concat_states, axis=0), np.mean(concat_actions, axis=0), np.std(concat_actions, axis=0)
       
if __name__ == "__main__":
    device = "cuda:1"  # Set the device to GPU cuda:1
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='default')
    args = parser.parse_args()

    # Initialize environment and model
    env_name = 'Pendulum-v1'
    # env_name = 'CAcrobat'
    
    if env_name == 'CAcrobat':
        env = ContinuousAcrobotEnv()
    else:
        env = gym.make(env_name)
    
    if AGENT == 'TD3':
        policy_name = 'policy/pendulum/Pi_3.pkl'  # You can change this based on the policy you're using
        # policy_name = 'policy/CAcrobat/TD3_CAcrobat_0_t25k.pkl'
        policy = TD3Policy(policy_name)
        
    
    # Generate D4RL-style trajectories
    num_episodes = 1000  #
    T = 256  # Define timesteps per episode
    trajectories = generate_trajectories_d4rl(env, policy, num_episodes, T)
    state_mean, state_std, action_mean, action_std = evaluate_trajectories(trajectories)

    # Set up a descriptive dataset name
    dataset_name = f'{args.name}'

    # Ensure 'dataset/' directory exists
    os.makedirs(f'dataset/{dataset_name}', exist_ok=True)

    # Save each component of the trajectories with detailed names
    np.save(f'dataset/{dataset_name}/observations.npy', np.concatenate([traj['observations'] for traj in trajectories]).reshape(-1, env.observation_space.shape[0]))
    np.save(f'dataset/{dataset_name}/actions.npy', np.concatenate([traj['actions'] for traj in trajectories]).reshape(-1, env.action_space.shape[0]))
    np.save(f'dataset/{dataset_name}/rewards.npy', np.concatenate([traj['rewards'] for traj in trajectories]))
    np.save(f'dataset/{dataset_name}/terminals.npy', np.concatenate([traj['terminals'] for traj in trajectories]))
    #save the mean and std too in json format
    json.dump({'state_mean': state_mean.tolist(), 'state_std': state_std.tolist(),
               'action_mean': action_mean.tolist(), 'action_std': action_std.tolist()}, open(f'dataset/{dataset_name}/normalization.json', 'w'))

    print(f'Successfully saved dataset: {dataset_name} in dataset/')
