import argparse
import pickle
import numpy as np
from tqdm import tqdm
import os
import sys
import torch
from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3 import PPO, SAC
import matplotlib.pyplot as plt
from matplotlib import cm


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../../..'))

from opelab.examples.simple.env import Custom2DEnv, EpsilonPolicy
from opelab.core.policy import MixturePolicy, PPOPolicy, SACPolicy
from opelab.core.baselines.diffusion.temporal import TemporalUnet
from opelab.core.baselines.diffusion.diffusion import GaussianDiffusion

def cycle(dl):
    while True:
        for data in dl:
            yield data

def roll_out(env, policy, T, num_trajectory, initial_state=None):
    SASR = []
    for _ in tqdm(range(num_trajectory)):
        sasr = []
        obs = env.reset()
        if initial_state is not None:
            env.set_state(initial_state)
            obs = initial_state
        for t in range(T):
            action = policy.sample()
            ns, action_with_noise = env.step(action)
            sasr.append((obs, action_with_noise, ns, 0))  # We don't need rewards for training the diffusion model
            obs = ns
        SASR.append(sasr)
    return SASR

@torch.no_grad
def get_log_likelihood(T, D, behavior_policy, diffusion_model, state_dim, windowed):
    cond = None
    if windowed:
        cond = {0: torch.zeros(1, state_dim).to(device)}
    dataset = diffusion_model.conditional_sample((100, T, state_dim + 1), cond).trajectories.detach()
    likelihoods = []
    dataset = unnormalize(dataset)
    
    for i in range(len(dataset)):
        s = dataset[i, :, :state_dim]
        a = dataset[i, :, state_dim:]
        log_prob = behavior_policy.log_prob(s, a)
        log_likelihood = log_prob.sum() / T
        likelihoods.append(log_likelihood.item())
    print(f'Log Likelihood of generated trajectories under train Policy: {np.mean(likelihoods)}')

def normalize(data):
    return data

def unnormalize(data):
    return data

def get_dataloader(data, batch_size, T, w, windowed):
    """
    Create a dataloader that provides either full trajectories or chunks of size w.
    Args:
        data: List of trajectories.
        batch_size: The size of the mini-batches.
        T: Total length of the trajectories.
        w: The window size (chunk length).
        windowed: If True, train on windows of size w; if False, train on full trajectories.
    """
    new_dataset = []
    
    for trajectory in data:
        if windowed:
            # Create sliding windows of size w over the trajectory
            for start in range(T - w + 1):
                chunk = []
                for t in range(start, start + w):
                    chunk.append((trajectory[t][0]).tolist() + [trajectory[t][1]])  # state + action
                new_dataset.append(chunk)
        else:
            # Add the full trajectory as a single unit
            full_trajectory = []
            for t in range(T):
                full_trajectory.append((trajectory[t][0]).tolist() + [trajectory[t][1]])  # state + action
            new_dataset.append(full_trajectory)
    
    new_dataset = torch.tensor(new_dataset, dtype=torch.float32)
    print(torch.mean(new_dataset))
    new_dataset = normalize(new_dataset)
    print(torch.mean(new_dataset))
    dataset = TensorDataset(new_dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def train(T, D, epoch, trainstep, accumulate, dataloader, diffusion_model, optimizer, behavior_policy, scheduler=None, state_dim=2, windowed=False):
    get_log_likelihood(T, D, behavior_policy, diffusion_model, state_dim, windowed)
    for i in range(epoch):
        loss_epoch = 0
        for step in tqdm(range(trainstep)):
            for _ in range(accumulate):
                batch = next(dataloader)[0]
                batch = batch.to(device)
                
                cond = None
                if windowed:
                    cond = {0: batch[:, 0, :state_dim]}

                loss, infos = diffusion_model.loss(batch, cond=cond)
                loss = loss / accumulate
                loss_epoch += loss.item()
                loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()
        print(f'epoch {i} loss: {loss_epoch / trainstep:.4f}), lr: {optimizer.param_groups[0]["lr"]}')
        get_log_likelihood(T, D, behavior_policy, diffusion_model, state_dim, windowed)

def visualize_trajectories(trajectories, T, name):
    """Visualize the sampled trajectories."""
    plt.style.use('ggplot')
    plt.figure(figsize=(8, 8))
    for trajectory in trajectories[:32]:
        x, y = trajectory[:, 0], trajectory[:, 1]
        colors = cm.binary(np.linspace(0.2, 1.0, len(x)))
        for i in range(len(x) - 1):
            plt.plot(x[i:i + 2], y[i:i + 2], lw=2, alpha=0.7, color=colors[i])

    for spine in plt.gca().spines.values(): 
        spine.set_edgecolor('black') 
        spine.set_linewidth(2)
    plt.gca().tick_params(axis='both', which='both', length=0)

    plt.xlabel('X', fontsize=36) 
    plt.ylabel('Y', fontsize=36)
    plt.gca().set_xticklabels([]) 
    plt.gca().set_yticklabels([])
    plt.grid(True)
    plt.xlim(-1, 5.5) 
    plt.ylim(-1, 5.5)
    plt.savefig(os.path.join(current_dir, 'outputs', f'training_{name}.pdf'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--T', type=int, default=128, help='timesteps')
    parser.add_argument('--D', type=int, default=128, help='diffusion steps')
    parser.add_argument('--epoch', type=int, default=60)
    parser.add_argument('--trainstep', type=int, default=2000)
    parser.add_argument('--accumulate', type=int, default=1)
    parser.add_argument('--rollouts', type=int, default=10000, help='number of rollouts')
    parser.add_argument('--out', type=str)
    parser.add_argument('--w', type=int, default=8, help='window size for trajectory chunks')
    parser.add_argument('--windowed', action='store_true', help='train on windows of size w instead of full trajectories')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = Custom2DEnv(K=0.05)

    # behavior data rollouts
    behavior_policy = EpsilonPolicy(epsilon=np.pi/4, sigma=0.2)
    b2 = EpsilonPolicy(epsilon=-np.pi/4, sigma=0.2)
    data1 = roll_out(env, behavior_policy, args.T, args.rollouts)
    data2 = roll_out(env, b2, args.T, args.rollouts, initial_state=np.array([0.0, 4.5]))
    data = data1 + data2 
 
    # Use T for full trajectories and w for windowed training
    horizon = args.w if args.windowed else args.T

    temporal_model = TemporalUnet(horizon=horizon, transition_dim=3).to(device)
    diffusion_model = GaussianDiffusion(
        model=temporal_model,
        horizon=horizon,
        observation_dim=2,
        action_dim=1,
        n_timesteps=args.D,
        policy=None,
        unnormalizer=unnormalize,
        normalizer=normalize
    ).to(device)

    # Adjust the dataloader based on windowed mode
    dataloader = cycle(get_dataloader(data, 32, args.T, args.w, args.windowed))

    test_samples = []
    for i in range(5):
        t = next(dataloader)[0]
        test_samples.append(t)
    test_samples = torch.cat(test_samples, dim=0)

    # Adjust visualization to reflect whether we're training on windows or full trajectories
    name = 'windowed' if args.windowed else 'full'
    visualize_trajectories(test_samples, horizon, name)

    optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    train(horizon, args.D, args.epoch, args.trainstep, args.accumulate, dataloader, diffusion_model, optimizer, behavior_policy=behavior_policy, scheduler=scheduler, windowed=args.windowed)

    path = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(path, 'diffusion', f'T{args.T}D{args.D}'), exist_ok=True)
    torch.save(diffusion_model.state_dict(), os.path.join(path, 'diffusion', f'T{args.T}D{args.D}', f'{args.out}.pth'))
