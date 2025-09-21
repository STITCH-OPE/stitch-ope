import os
import sys
import gym
import d4rl
import torch
import numpy as np
import argparse
import pickle
from tqdm import tqdm
from typing import List, Union, Optional, Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../../..'))

from opelab.core.policy import D4RLSACPolicy, DiffusionPolicy
from cleandiffuser.nn_condition import PearceObsCondition
from cleandiffuser.nn_diffusion import PearceMlp
from cleandiffuser.diffusion.diffusionsde import DiscreteDiffusionSDE


def find_policy_files(policy_dir):
    """Find all .pkl files in the given directory"""
    if not os.path.exists(policy_dir):
        raise FileNotFoundError(f"Policy directory {policy_dir} does not exist")
    
    policy_files = []
    for file in os.listdir(policy_dir):
        if file.endswith('.pkl'):
            policy_files.append(os.path.join(policy_dir, file))
    
    if not policy_files:
        raise ValueError(f"No .pkl files found in {policy_dir}")
    
    return policy_files


def parse_args():
    parser = argparse.ArgumentParser(description="Train a diffusion policy with CleanDiffuser.")
    parser.add_argument("--env", type=str, default="Hopper-v2", help="Gym environment name for training and evaluation")
    parser.add_argument("--steps", type=int, default=100000, help="Total training steps")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--num_checkpoints", type=int, default=10, help="Number of spaced checkpoints")
    parser.add_argument("--save_path", type=str, default="policy/hopper2", help="Path to save models")
    parser.add_argument("--dataset_size", type=int, default=1000000, help="Number of transitions to generate for each policy dataset")
    parser.add_argument("--policy_dir", type=str, default=None, 
                        help="Directory containing policy .pkl files to load.")
    return parser.parse_args()


def setup_environment(env_name):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    return env, obs_dim, act_dim


def generate_dataset_for_policy(env, policy_file: str, size: int, device):
    """Generate a dataset by rolling out a D4RLSACPolicy"""
    print(f"Generating dataset of size {size} using policy from: {policy_file}")
    
    # Load D4RLSACPolicy from pkl file
    d4rl_policy = D4RLSACPolicy(policy_file)
    d4rl_policy.to(device)
    
    observations = []
    actions = []
    rewards = []
    next_observations = []
    terminals = []
    
    total_transitions = 0
    
    progress_bar = tqdm(total=size, desc="Collecting transitions")
    
    while total_transitions < size:
        obs = env.reset()
        done = False
        
        while not done and total_transitions < size:
            action = d4rl_policy.sample(obs)
            next_obs, reward, done, _ = env.step(action)
            
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            next_observations.append(next_obs)
            terminals.append(done)
            
            obs = next_obs
            total_transitions += 1
            progress_bar.update(1)
            
    progress_bar.close()
    
    # Convert to numpy arrays
    dataset = {
        'observations': np.array(observations[:size]),
        'actions': np.array(actions[:size]),
        'rewards': np.array(rewards[:size]),
        'next_observations': np.array(next_observations[:size]),
        'terminals': np.array(terminals[:size])
    }
    
    print(f"Generated dataset with {len(dataset['observations'])} transitions")
    return dataset


def create_model(obs_dim, act_dim, device):
    nn_diffusion = PearceMlp(act_dim, To=1, emb_dim=64, hidden_dim=256, timestep_emb_type="positional")
    nn_condition = PearceObsCondition(obs_dim, emb_dim=64, flatten=True, dropout=0.0)

    actor = DiscreteDiffusionSDE(
        nn_diffusion, nn_condition, predict_noise=True, optim_params={"lr": 3e-4},
        x_max=+1. * torch.ones((1, act_dim)),
        x_min=-1. * torch.ones((1, act_dim)),
        diffusion_steps=32, ema_rate=0.9999, device=device
    )
    return actor


def train_and_save_policy(actor, dataset, args, device, model_name="diffusion_final"):
    """Train a diffusion policy and save it with the given name"""
    print(f"Training policy '{model_name}'...")
    
    save_interval = args.steps // args.num_checkpoints
    size = len(dataset['observations'])
    
    os.makedirs(args.save_path, exist_ok=True)
    avg_loss = 0.

    progress_bar = tqdm(range(args.steps), desc=f"Training {model_name}")
    
    for t in progress_bar:
        idx = np.random.randint(0, size, (args.batch_size,))
        obs = torch.tensor(dataset['observations'][idx], device=device).float()
        act = torch.tensor(dataset['actions'][idx], device=device).float()

        loss_info = actor.update(act, obs)
        avg_loss += loss_info["loss"]

        if (t + 1) % 1000 == 0:
            progress_bar.set_postfix({'Loss': f'{avg_loss / 1000:.4f}'})
            avg_loss = 0.

    final_model_path = os.path.join(args.save_path, f"{model_name}.pt")
    actor.save(final_model_path)
    print(f"Final model saved: {final_model_path}")
    
    return actor


def evaluate(actor, env_name, act_dim, device):
    print("Evaluating trained model...")
    
    actor.eval()
    num_episodes = 10
    total_rewards = []
    
    env_eval = gym.make(env_name)
    
    for episode in range(num_episodes):
        obs = env_eval.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Convert single observation to tensor
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            # Create prior noise for a single sample
            prior = torch.zeros((1, act_dim), device=device)
            
            # Sample action
            act, _ = actor.sample(
                prior, solver="ddpm", n_samples=1, sample_steps=5,
                temperature=0.5, w_cfg=1.0,
                condition_cfg=obs_tensor
            )
            
            # Extract the single action
            action = act.cpu().numpy().squeeze(0)
            
            # Step environment
            obs, reward, done, _ = env_eval.step(action)
            episode_reward += reward
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode+1}/{num_episodes}: Reward = {episode_reward:.2f}")
    
    mean_reward = np.mean(total_rewards)
    print(f"Mean evaluation reward over {num_episodes} episodes: {mean_reward:.2f}")
    env_eval.close()
    
    return mean_reward


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env, obs_dim, act_dim = setup_environment(args.env)
    
    # Find all pkl policy files in the directory
    policy_files = find_policy_files(args.policy_dir)
    print(f"Found {len(policy_files)} policy files in {args.policy_dir}")
    
    # Process each policy file separately
    for i, policy_file in enumerate(policy_files):
        policy_name = os.path.splitext(os.path.basename(policy_file))[0]
        print(f"\n[{i+1}/{len(policy_files)}] Processing policy: {policy_name}")
        
        # Generate a dataset using the D4RLSACPolicy
        print(f"Generating dataset for {policy_name}...")
        dataset = generate_dataset_for_policy(env, policy_file, args.dataset_size, device)
        
        # Create and train a diffusion policy on this specific dataset
        print(f"Training diffusion policy based on {policy_name}...")
        actor = create_model(obs_dim, act_dim, device)
        actor = train_and_save_policy(actor, dataset, args, device, model_name=policy_name)
        
        # Evaluate if this is the last model
        if i == len(policy_files) - 1:
            evaluate(actor, args.env, act_dim, device)
            
        print(f"Finished processing {policy_name}\n")