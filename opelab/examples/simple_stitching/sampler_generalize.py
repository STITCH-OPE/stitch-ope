import argparse
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../../..'))

from opelab.core.baselines.diffusion.temporal import TemporalUnet
from opelab.core.baselines.diffusion.diffusion import GaussianDiffusion
from opelab.examples.simple.env import EpsilonPolicy, Custom2DEnv, SumPolicy


def visualize_trajectories(trajectories, T, name):
    """Visualize the sampled trajectories."""
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(8, 8))

    for trajectory in trajectories:
        x, y = trajectory[:, 0], trajectory[:, 1]
        colors = cm.Reds(np.linspace(0.2, 1.0, len(x)))
        for i in range(len(x) - 1):
            plt.plot(x[i:i + 2], y[i:i + 2], lw=2, alpha=0.7, color=colors[i])
    
    #I want the arrows to be on top of the trajectories
    theta = np.pi/4
    lf = 1.0
    plt.arrow(0, 0, lf*np.cos(theta), lf*np.sin(theta), 
              head_width=0.2, head_length=0.05, width=0.08, fc='gray', ec='gray', zorder=10)
    plt.arrow(0, 0, lf*np.cos(-theta), lf*np.sin(-theta), 
              head_width=0.2, head_length=0.05, width=0.08, fc='gray', ec='gray', zorder=10)
    
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
    plt.savefig(os.path.join(current_dir, 'outputs', f'samples_{name}.pdf'))

def generate_trajectories_stepwise(diffusion_model, initial_state, num_samples, horizon, generation_horizon, device):
    current_state = initial_state.repeat(num_samples, 1)  # Repeat for the number of samples
    sampled_trajectories = None
    print(f"Generating {num_samples} trajectories with {generation_horizon} steps, each with {horizon} steps")
    for t in range(generation_horizon // horizon):
        cond = {
            0: current_state,  # Condition on the current state
        }        
        next_step = diffusion_model.conditional_sample(
            (num_samples, horizon, 3), cond=cond, guided=True, action_scale=0.5, use_neg_grad=False).trajectories.detach()
        normalized_next = next_step.clone()
        
        if t == 0:
            sampled_trajectories = next_step
        else:
            sampled_trajectories = torch.cat([sampled_trajectories, next_step], dim=1)
        current_state = normalized_next[:, -1, :2]  

    return sampled_trajectories

def visualize_ground_truth_trajectories(env, policy, num_samples, horizon):
    all_trajectories = []
    init_ys = np.linspace(0, 5, num_samples)
    for y in init_ys:
        env.reset()  
        env.state = np.array([0.0, y])
        trajectory_pos = [env.state.copy()]
        for _ in range(horizon):
            action = policy.sample(env.state.copy()[0])
            state, _ = env.step(action)
            trajectory_pos.append(state.copy())
        all_trajectories.append(np.array(trajectory_pos))
    trajectories = np.array(all_trajectories)
    
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(8, 8))

    for trajectory in trajectories:
        x, y = trajectory[:, 0], trajectory[:, 1]
        colors = cm.Blues(np.linspace(0.2, 1.0, len(x)))
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
    plt.savefig(os.path.join(current_dir, 'outputs', f'expected_behavior_generalize.pdf'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--w', type=int, default=8, help='diffusion timesteps')
    parser.add_argument('--windowed', action='store_true', help='evaluate on windows of size w instead of full trajectories')
    parser.add_argument('--T', type=int, default=128, help='timesteps')
    parser.add_argument('--D', type=int, default=128, help='diffusion steps')
    parser.add_argument('--model_path', type=str, default='diffusion/T128D128/windowed.pth', help='path to the trained diffusion model .pth file')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    behavior_policy = EpsilonPolicy(epsilon=np.pi/4, sigma=0.2)
    target_policy = EpsilonPolicy(epsilon=-np.pi/4, sigma=0.2)
    target_policy = SumPolicy(behavior_policy, target_policy, 2)

    env = Custom2DEnv(K=0.05)

    # Use T for full trajectories and w for windowed training
    horizon = args.w if args.windowed else args.T
    
    # Load the trained diffusion model
    temporal_model = TemporalUnet(horizon=horizon, transition_dim=3).to(device)
    diffusion_model = GaussianDiffusion(
        model=temporal_model,
        horizon=horizon,
        observation_dim=2,
        action_dim=1,
        n_timesteps=args.D,
        policy=target_policy,
        unnormalizer=lambda x: x,
        normalizer=lambda x: x
    ).to(device)

    # Load the model state
    model_state_path = os.path.join(current_dir, args.model_path)
    diffusion_model.load_state_dict(torch.load(model_state_path))
    diffusion_model.eval()

    # Sample trajectories from the diffusion model
    trajectories_all = []
    for y in np.linspace(0.0, 5.0, 8):
        initial_state = torch.tensor([0.0, y], device=device).unsqueeze(0)   
        num_samples = 1
        cond = {
            0: torch.tensor([0.0, y]),  # Initial state
        }
        if args.windowed:
            sampled_trajectories = generate_trajectories_stepwise(
                diffusion_model, initial_state, num_samples, args.w, args.T, device)
        else:
            sampled_trajectories = diffusion_model.conditional_sample(
                (num_samples, args.T, 3), cond=cond, guided=True, action_scale=0.5, use_neg_grad=False).trajectories.detach()
        
        trajectories_np = sampled_trajectories.cpu().numpy()
        trajectories_all.append(trajectories_np)
    trajectories_np = np.concatenate(trajectories_all, axis=0)
    
    # Extract the (x, y) positions from the state dimension
    extracted_trajectories = []
    for trajectory in trajectories_np:
        extracted_trajectory = trajectory[:, :2]  # Extract (x, y)
        extracted_trajectories.append(extracted_trajectory)
    
    # Visualize the sampled trajectories
    name = 'windowed_generalize' if args.windowed else 'full_generalize'
    visualize_trajectories(extracted_trajectories, args.T, name)
    visualize_ground_truth_trajectories(env, target_policy, 8, args.T)
