import argparse
import gym
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import sys
import os
from tqdm import tqdm
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../../..'))

from opelab.core.baselines.diffusion.temporal import TemporalUnet
from opelab.core.baselines.diffusion.diffusion import GaussianDiffusion
from opelab.core.task import ContinuousAcrobotEnv

deviceno = torch.cuda.current_device()
device = torch.device(f'cuda:{deviceno}' if torch.cuda.is_available() else 'cpu')
print(f'using device: {device}')


def load_generated_dataset(dataset_name):
    observations = np.load(f'dataset/{dataset_name}/observations.npy')
    actions = np.load(f'dataset/{dataset_name}/actions.npy')
    rewards = np.load(f'dataset/{dataset_name}/rewards.npy')
    terminals = np.load(f'dataset/{dataset_name}/terminals.npy')
    return observations, actions, rewards, terminals

def load_mean_std(dataset_name):
    normalization_path = f'dataset/{dataset_name}/normalization.json'
    with open(normalization_path, 'r') as f:
        normalization = json.load(f)
    mean_s = torch.tensor(normalization['state_mean'], dtype=torch.float32)
    std_s = torch.tensor(normalization['state_std'], dtype=torch.float32)
    mean_a = torch.tensor(normalization['action_mean'], dtype=torch.float32)
    std_a = torch.tensor(normalization['action_std'], dtype=torch.float32)
    mean = torch.cat([mean_s, mean_a])
    std = torch.cat([std_s, std_a])
    return mean, std

# Compute mean and std based on the dataset
def compute_mean_std(dataset):
    mean = dataset.mean(axis=(0,1))
    std = dataset.std(axis=(0,1))
    return mean, std

def normalize(data, mean, std):
    return (data - mean) / std

def unnormalize(data, mean, std):
    return data * std + mean

def get_dataloader(episodic_data, batch_size):
    combined_dataset = CustomDataset(episodic_data)
    dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def cycle(dl):
    while True:
        for data in dl:
            yield data

def generate_and_print_trajectories(diffusion_model, T, unnormalize_fn, cond=True):
    num_samples = 1  # Adjust the number of samples as needed
    #generated_trajectories = diffusion_model.p_sample_loop((num_samples, T , state_dim + action_dim), scale=0)[0]
    # x_position = np.random.uniform(low=-.005, high=.005, size=(1,))
    
    state = env.reset()
    state = torch.tensor(np.concatenate((state, np.zeros(action_dim))), device=device)  # Add zero delta
    state = normalize_fn(state)
    state = state[:state_dim]

    if cond:
        cond = {0:state}
    else:
        cond = None
    
    generated_trajectories = diffusion_model.conditional_sample((num_samples, T , state_dim + action_dim), cond)[0]
    unnormalized_trajectories = unnormalize_fn(generated_trajectories)


    # Print the first element of the first time step of each trajectory
    for i in range(2):
        print(f'Trajectory {i} element : {unnormalized_trajectories[0, i, 0]}')


# Extract valid trajectories from the dataset
def extract_valid_trajectories_v2(observations, actions, rewards, terminals, T, S, max_trajectories=None):
    trajectories = []
    num_trajectories = 0
    start_idx = 0
    num_samples = len(terminals)

    with tqdm(total=num_samples - T, desc="Extracting Trajectories", unit="step") as pbar:
        while start_idx < num_samples - T:
            end_idx = start_idx + T
            if np.any(terminals[start_idx:end_idx]):
                start_idx += 1
                pbar.update(1)
                continue

            trajectory = {
                'observations': observations[start_idx:end_idx],
                'actions': actions[start_idx:end_idx],
                'rewards': rewards[start_idx:end_idx],
                'terminals': terminals[start_idx:end_idx],
            }
            trajectories.append(trajectory)
            num_trajectories += 1

            if max_trajectories is not None and num_trajectories >= max_trajectories:
                break

            start_idx += S
            pbar.update(S)

    # Convert the trajectories into a state-action array
    state_action_array = np.zeros((len(trajectories), T, observations.shape[1] + actions.shape[1]), dtype=np.float32)
    for traj_idx, traj in enumerate(trajectories):
        for i in range(T):
            observation = traj['observations'][i]
            action = traj['actions'][i]
            state_action_array[traj_idx, i] = np.concatenate([observation, action])

    return state_action_array

def extract_valid_trajectories_v3(observations, actions, rewards, terminals, T, S, max_trajectories=None):
    """
    Extract trajectories of length T from the data. If a terminal is encountered within a window,
    include data up to the terminal and pad the remaining steps with zeros.

    Parameters:
    - observations: np.array of shape [num_steps, obs_dim]
    - actions: np.array of shape [num_steps, action_dim]
    - rewards: np.array of shape [num_steps]
    - terminals: np.array of shape [num_steps], binary flags indicating terminal steps
    - T: int, trajectory length
    - S: int, step size for sliding window
    - max_trajectories: int or None, maximum number of trajectories to extract

    Returns:
    - state_action_array: np.array of shape [num_trajectories, T, obs_dim + action_dim]
    """
    trajectories = []
    num_trajectories = 0
    start_idx = 0
    num_samples = len(terminals)

    obs_dim = observations.shape[1]
    action_dim = actions.shape[1]

    with tqdm(total=num_samples - T, desc="Extracting Trajectories", unit="step") as pbar:
        while start_idx < num_samples - T:
            end_idx = start_idx + T
            window_terminals = terminals[start_idx:end_idx]
            terminal_indices = np.where(window_terminals)[0]
            
            if len(terminal_indices) == 0:
                # No terminal in the window, include the entire window
                traj_obs = observations[start_idx:end_idx]
                traj_actions = actions[start_idx:end_idx]
                traj_rewards = rewards[start_idx:end_idx]
                traj_terminals = terminals[start_idx:end_idx]
            else:
                # Terminal found within the window
                first_terminal = terminal_indices[0]
                traj_length = first_terminal + 1  # Include the terminal step

                # Extract data up to the terminal
                traj_obs = observations[start_idx:start_idx + traj_length]
                traj_actions = actions[start_idx:start_idx + traj_length]
                traj_rewards = rewards[start_idx:start_idx + traj_length]
                traj_terminals = terminals[start_idx:start_idx + traj_length]

                # Calculate how many steps to pad
                pad_length = T - traj_length
                if pad_length > 0:
                    # Create padding arrays filled with zeros
                    pad_observations = np.zeros((pad_length, obs_dim), dtype=observations.dtype)
                    pad_actions = np.zeros((pad_length, action_dim), dtype=actions.dtype)
                    pad_rewards = np.zeros(pad_length, dtype=rewards.dtype)
                    pad_terminals = np.zeros(pad_length, dtype=terminals.dtype)

                    # Concatenate the actual data with padding
                    traj_obs = np.concatenate([traj_obs, pad_observations], axis=0)
                    traj_actions = np.concatenate([traj_actions, pad_actions], axis=0)
                    traj_rewards = np.concatenate([traj_rewards, pad_rewards], axis=0)
                    traj_terminals = np.concatenate([traj_terminals, pad_terminals], axis=0)

            # Combine observations and actions into a single array
            state_action = np.concatenate([traj_obs, traj_actions], axis=1)  # Shape: [T, obs_dim + action_dim]
            trajectories.append(state_action)
            num_trajectories += 1
            pbar.update(S)

            # Check if we've reached the maximum number of trajectories
            if max_trajectories is not None and num_trajectories >= max_trajectories:
                break

            # Move the window by step size S
            start_idx += S

    # Convert the list of trajectories into a NumPy array
    state_action_array = np.stack(trajectories, axis=0)  # Shape: [num_trajectories, T, obs_dim + action_dim]
    return state_action_array

def train(T, D, epoch, trainstep, accumulate, dataloader, diffusion_model, optimizer, scheduler=None, unnormalize_fn=None, cond=True):
    n_epochs = epoch
    n_train_steps = trainstep
    gradient_accumulate_every = accumulate

    for i in range(n_epochs):
        loss_epoch = 0
        for step in tqdm(range(n_train_steps)):
            for j in range(gradient_accumulate_every):
                batch = next(dataloader)
                batch = batch.to(device)

                conds = batch[:, 0, :state_dim]
                if cond:
                    conds = {0: conds}
                else:
                    conds = None

                loss, infos = diffusion_model.loss(batch, conds)
                loss = loss / gradient_accumulate_every
                loss_epoch += loss.item()
                loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()
        print(f'epoch {i} loss: {loss_epoch / n_train_steps:.4f}), lr: {optimizer.param_groups[0]["lr"]}')
        
        # Generate and print small batch of trajectories
        generate_and_print_trajectories(diffusion_model, T, unnormalize_fn, cond)

class CustomDataset(Dataset):
    def __init__(self, episodic_data):
        self.episodic_data = episodic_data

    def __len__(self):
        return len(self.episodic_data)

    def __getitem__(self, idx):
        return self.episodic_data[idx]

# Load, preprocess and extract valid trajectories from the dataset
def load_and_preprocess_generated_dataset(dataset_name, T, S):
    # Load your dataset
    observations, actions, rewards, terminals = load_generated_dataset(dataset_name)
    
    # Extract valid trajectories using extract_valid_trajectories_v2
    episodic_data = extract_valid_trajectories_v2(observations, actions, rewards, terminals, T, S)
    
    return torch.tensor(episodic_data, dtype=torch.float32)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--T', type=int, default=32, help='timesteps')
    parser.add_argument('--D', type=int, default=256, help='diffusion steps')
    parser.add_argument('--S', type=int, default=1, help='stride')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--trainstep', type=int, default=5000)
    parser.add_argument('--accumulate', type=int, default=1)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--normalize', type=str, default='gaussian')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset to load')
    parser.add_argument('--uncond', action='store_false', dest='cond' ,help='Unconditional diffusion')
    parser.add_argument('--env_name', type=str, default='Pendulum-v1')

    args = parser.parse_args()

    if args.env_name == 'CAcrobat':
        env = ContinuousAcrobotEnv()
    else:
        env = gym.make(args.env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print(f'state_dim = {state_dim}')
    print(f'action_dim = {action_dim}')

    print(f'conditioned: {args.cond}')

    # Load and preprocess the dataset
    episodic_data = load_and_preprocess_generated_dataset(args.dataset, args.T, args.S)

    # Normalize the dataset based on the dataset itself
    if args.normalize == 'gaussian':    
        #mean, std = compute_mean_std(episodic_data)

        mean,std = load_mean_std(args.dataset)
        print(f'mean: {mean}')
        print(f'std: {std}')
        
        normalized_data = normalize(episodic_data, mean, std)

        normalize_fn = lambda data: normalize(data, mean.to(device), std.to(device))
        unnormalize_fn = lambda data: unnormalize(data, mean.to(device), std.to(device))
    else:
        normalized_data = episodic_data
        normalize_fn = None
        unnormalize_fn = None

    dataloader = cycle(get_dataloader(normalized_data, 128))

    # Define your temporal diffusion model
    temporal_model = TemporalUnet(horizon=args.T, transition_dim=state_dim + action_dim, dim_mults=(1,)).to(device)
    diffusion_model = GaussianDiffusion(
        model=temporal_model,
        horizon=args.T,
        observation_dim=state_dim,
        action_dim=action_dim,
        n_timesteps=args.D,
        normalizer=normalize_fn,
        unnormalizer=unnormalize_fn
    ).to(device)

    optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2 * args.epoch)

    # Train the model
    train(args.T, args.D, args.epoch, args.trainstep, args.accumulate, dataloader, diffusion_model, optimizer, scheduler, unnormalize_fn, args.cond)

    Path_name = args.env_name.split('-')[0].lower()
    # Save the trained model
    path = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(path, Path_name, f'T{args.T}D{args.D}'), exist_ok=True)
    torch.save(diffusion_model.state_dict(), os.path.join(path, Path_name, f'T{args.T}D{args.D}', f'{args.out}.pth'))
