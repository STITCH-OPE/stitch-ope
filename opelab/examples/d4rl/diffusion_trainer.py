import argparse
import gym
import d4rl
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
from opelab.core.baselines.diffusion.helpers import EMA

device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'using device: {device}')

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

def generate_and_print_trajectories(env, diffusion_model, T, unnormalize_fn, cond=True, atanh=False):
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
        print(f'Trajectory State {i} element : {unnormalized_trajectories[0, i, :state_dim]}')
        print(f'Trajectory Action {i} element : {unnormalized_trajectories[0, i, state_dim:]}')
        


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

def train(env, T, D, epoch, trainstep, accumulate, dataloader, diffusion_model, ema, optimizer, scheduler=None, unnormalize_fn=None, cond=True, atanh=False):
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
            ema.update(diffusion_model)
                        
        if scheduler is not None:
            scheduler.step()
        print(f'epoch {i} loss: {loss_epoch / n_train_steps:.4f}), lr: {optimizer.param_groups[0]["lr"]}')
        
        generate_and_print_trajectories(env, ema.ema_model, T, unnormalize_fn, cond, atanh)

        
class CustomDataset(Dataset):
    def __init__(self, episodic_data):
        self.episodic_data = episodic_data

    def __len__(self):
        return len(self.episodic_data)

    def __getitem__(self, idx):
        return self.episodic_data[idx]

def load_and_preprocess_generated_dataset_d4rl(dataset_name, T, S, atanh=False):
    dataset = gym.make(dataset_name)
    dataset = dataset.get_dataset()
    
    observations = dataset['observations']
    actions = dataset['actions']
    rewards = dataset['rewards']
    terminals = dataset['terminals']
    timeouts = dataset['timeouts']
    end_idx = dataset['timeouts'] | dataset['terminals']
    num_samples = len(terminals)
    
    
    #The boundary has been assumed to be [-1, 1] for both observations and actions, this is not always the case (TODO: fix this)
    #check if there is an action out of bounds
    if np.any(actions > 1) or np.any(actions < -1):
        print('*'*50)
        print('Warning: Some actions are out of bounds!')
        print('*'*50)
        
    if atanh:
        epsilon = 1e-6
        actions = np.clip(actions, -1 + epsilon, 1 - epsilon)
        actions = np.arctanh(actions)

    episodic_data = extract_valid_trajectories_v2(observations, actions, rewards, end_idx, T, S)        
    
    mean_state, std_state = np.mean(observations, axis=0), np.std(observations, axis=0)
    mean_action, std_action = np.mean(actions, axis=0), np.std(actions, axis=0)
    mean, std = np.concatenate((mean_state, mean_action)), np.concatenate((std_state, std_action))
    print(f'mean: {mean}')
    print(f'std: {std}')    
    
    output_dict = {
        'mean': mean,
        'std': std,
        'episodic_data': episodic_data
    }
    
    return output_dict
    
    

Path_name = 'diffusion'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Hopper-v2', help='environment name')
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
    parser.add_argument('--atanh', action='store_true', help='Use atanh for action normalization')

    args = parser.parse_args()

    print(f'conditioned: {args.cond}')
    print(f'atanh: {args.atanh}')
    
    env_name = args.env
    
    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print(f'state_dim = {state_dim}')
    print(f'action_dim = {action_dim}')

    data = load_and_preprocess_generated_dataset_d4rl(args.dataset, args.T, args.S, args.atanh)
    episodic_data = data['episodic_data']
    mean = data['mean']
    std = data['std']
        
    if args.normalize == 'gaussian':    
        
        normalized_data = normalize(episodic_data, mean, std)
        mean = torch.tensor(mean, device=device)
        std = torch.tensor(std, device=device)
        normalize_fn = lambda data: normalize(data, mean, std)
        unnormalize_fn = lambda data: unnormalize(data, mean, std)
    else:
        normalized_data = episodic_data
        normalize_fn = None
        unnormalize_fn = None

    dataloader = cycle(get_dataloader(normalized_data, 128))

    temporal_model = TemporalUnet(horizon=args.T, transition_dim=state_dim + action_dim, dim_mults=(1,), attention=False).to(device)
    diffusion_model = GaussianDiffusion(
        model=temporal_model,
        horizon=args.T,
        observation_dim=state_dim,
        action_dim=action_dim,
        n_timesteps=args.D,
        normalizer=normalize_fn,
        unnormalizer=unnormalize_fn,
        predict_epsilon=True,
        loss_type='l2',
        clip_denoised=False,
        action_weight=1,
        loss_weights = None,
        loss_discount = 1

    ).to(device)
    
    ema = EMA(diffusion_model)
    
    optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2 * args.epoch)

    train(env, args.T, args.D, args.epoch, args.trainstep, args.accumulate, dataloader, diffusion_model, ema, optimizer, scheduler, unnormalize_fn, args.cond, args.atanh)

    env_base_name = env_name.split('-')[0]

    path = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(path, Path_name, env_base_name, f'T{args.T}D{args.D}'), exist_ok=True)
    torch.save(diffusion_model.state_dict(), os.path.join(path, Path_name, env_base_name, f'T{args.T}D{args.D}', f'{args.out}.pth'))
    torch.save(ema.ema_model.state_dict(), os.path.join(path, Path_name, env_base_name, f'T{args.T}D{args.D}', f'{args.out}_ema.pth'))
