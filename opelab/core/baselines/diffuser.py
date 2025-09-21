import numpy as np

import torch

from opelab.core.baseline import Baseline
from opelab.core.data import DataType
from opelab.core.policy import Policy
from opelab.core.baselines.diffusion.diffusion import GaussianDiffusion
from opelab.core.baselines.diffusion.temporal import TemporalUnet
from opelab.core.task import ContinuousAcrobotEnv
from opelab.examples.helpers import TanhBijector
import gym
import d4rl
import os

class Diffuser(Baseline):

    def __init__(self, T:int, D:int, num_samples:int, state_dim:int, action_dim:int, device, unnormalizer, normalizer, reward_fn, model_path, target_model, scale, behavior_model, env, guided=True, T_gen=768, is_terminated_fn=None, tanh_action=False, guidance_hyperparams=None, save=False, dim_mults=(1,2,4,8),attention=False, predict_epsilon=False) -> None:
        self.num_samples = num_samples
        self.T = T
        self.D = D
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.normalizer = normalizer
        self.unnormalizer = unnormalizer
        self.reward_fn = reward_fn
        self.scale = scale
        self.env = env
        self.env_name = env.spec.id
        self.env_min = env.action_space.low
        self.env_max = env.action_space.high
        self.guided = guided
        self.T_gen = T_gen
        self.is_terminated_fn = is_terminated_fn
        self.save = save
        
        if guided and guidance_hyperparams is None:
            print('******** No guidance hyperparameters provided. Using default values. ********')
            guidance_hyperparams = {
                'action_scale': 0.2,
                'state_scale': 0.01,
                'use_adaptive': True,
                'use_neg_grad': True,
                'neg_grad_scale': 0.2,
                'normalize_grad': True,
                'k_guide': 2,
                'use_action_grad_only': True,
                'clamp': True,
                'l_inf': 1,
                'ratio': 0.5
            }
            
        if is_terminated_fn is None:
            print('******** No termination function provided. Using infinite horizon case. ********')
        
        self.k_guide = guidance_hyperparams['k_guide']
        self.use_adaptive = guidance_hyperparams['use_adaptive']
        self.use_neg_grad = guidance_hyperparams['use_neg_grad']
        self.neg_grad_scale = guidance_hyperparams['neg_grad_scale']
        self.normalize_grad = guidance_hyperparams['normalize_grad']
        self.use_action_grad_only = guidance_hyperparams['use_action_grad_only']
        self.action_scale = guidance_hyperparams['action_scale']
        self.state_scale = 0
        self.clamp = guidance_hyperparams['clamp']
        self.l_inf = guidance_hyperparams['l_inf']
        self.ratio = guidance_hyperparams['ratio']
        
        self.tanh_action = tanh_action
        self.predict_epsilon = predict_epsilon
        


        temporal_model = TemporalUnet(
            horizon=T,
            transition_dim=state_dim + action_dim,
            attention=attention,
            dim_mults = dim_mults
        ).to(device)

        transform = TanhBijector() if tanh_action else None
        
        diffusion_model = GaussianDiffusion(
            model=temporal_model,
            horizon=T,
            observation_dim=state_dim,
            action_dim=action_dim,
            n_timesteps=D,
            normalizer=normalizer,
            unnormalizer=unnormalizer,
            gmode=tanh_action,
            transform=transform,
            action_weight = 5,
            clip_denoised = False,
            loss_discount = 1, 
            loss_type = 'l2',
            loss_weights = None,
            predict_epsilon = self.predict_epsilon,  
        )

        diffusion_model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
        diffusion_model.to(device)
        target_model.to(device)
        if behavior_model is not None:
            behavior_model.to(device)
        diffusion_model.policy=target_model
        diffusion_model.behavior_policy = behavior_model

        self.diffusion_model = diffusion_model
        self.diffusion_model.eval()

    def unsquash_action(self, action):
        return (action + 1) * (self.env_max - self.env_min) / 2 + self.env_min

    def squash_action(self, action):
        return 2 * (action - self.env_min) / (self.env_max - self.env_min) - 1

    def set_guidance_hyperparams(self, guidance_hyperparams):
        for key, value in guidance_hyperparams.items():
            setattr(self, key, value)
            
    def get_initial_state(self, env_name, device='cpu'):
        if env_name == 'ContinuousAcrobot':
            env = ContinuousAcrobotEnv()
        else:
            env = gym.make(env_name)
        initial_state_tensor = torch.zeros((self.num_samples, self.state_dim), device=device)
        for i in range(self.num_samples):
            initial_state = env.reset()
            initial_state_tensor[i] = torch.tensor(initial_state, dtype=torch.float32, device=device)
        return initial_state_tensor

    def evaluate(self, data: DataType, target: Policy, behavior: Policy, gamma: float = 1.0, reward_estimator=None) -> float:

        def is_terminated(state):
            if self.is_terminated_fn is not None:
                return self.is_terminated_fn(state)
            else:
                return False

        def generate_full_trajectory(diffusion_model, normalize_fn, unnormalize_fn, is_terminated_fn, initial_state, 
                                    state_dim, action_dim, batch_size=10, trajectory_max_length=32, mini_trajectory_size=8, device='cuda'):
            
            normalized_initial = normalize_fn(
                torch.cat([
                    initial_state, 
                    torch.zeros([self.num_samples, action_dim], device=device)
                ], dim=1)
            )
            normalized_initial = normalized_initial[:, :state_dim]

            alive_indices = torch.arange(0, batch_size, dtype=torch.long, device=device) 
            conditions = {0: normalized_initial}  
            all_trajectories = torch.zeros((batch_size, trajectory_max_length, state_dim + action_dim), device=device)
            end_indices = torch.full((batch_size,), trajectory_max_length, dtype=torch.long, device=device) 

            total_generated = 0  

            while alive_indices.numel() > 0 and total_generated < trajectory_max_length:
                current_batch_size = alive_indices.size(0)
                shape = (current_batch_size, mini_trajectory_size, state_dim + action_dim)
                
                samples = diffusion_model.conditional_sample(shape, conditions, guided=self.guided, action_scale=self.action_scale, state_scale=self.state_scale,
                                                            use_adaptive=self.use_adaptive, use_neg_grad=self.use_neg_grad, neg_grad_scale=self.neg_grad_scale,
                                                            normalize_grad=self.normalize_grad, k_guide=self.k_guide, use_action_grad_only=self.use_action_grad_only,
                                                            clamp=self.clamp, l_inf=self.l_inf, ratio=self.ratio)[0]
                samples = unnormalize_fn(samples)
                if self.tanh_action:
                    samples[:, :, state_dim:] = torch.tanh(samples[:, :, state_dim:])

                new_alive_indices = []
                new_local_alive_indices = []
                steps_added = min(trajectory_max_length - total_generated, mini_trajectory_size - 1)

                for local_idx, global_idx in enumerate(alive_indices):
                    trajectory = samples[local_idx]
                    for step_idx in range(mini_trajectory_size):
                        if total_generated + step_idx >= trajectory_max_length:
                            end_indices[global_idx] = trajectory_max_length
                            all_trajectories[global_idx, total_generated:] = trajectory[:step_idx]
                            break  # Stop if the total trajectory length is exceeded
                        if is_terminated_fn(trajectory[step_idx, :state_dim]):
                            end_indices[global_idx] = min(total_generated + step_idx, trajectory_max_length)
                            all_trajectories[global_idx, total_generated:total_generated + step_idx] = trajectory[:step_idx]
                            break
                    else:
                        steps_to_store = min(mini_trajectory_size - 1, trajectory_max_length - total_generated)
                        all_trajectories[global_idx, total_generated:total_generated + steps_to_store] = trajectory[:steps_to_store]
                        new_alive_indices.append(global_idx)
                        new_local_alive_indices.append(local_idx)

                alive_indices = torch.tensor(new_alive_indices, device=device)  

                if alive_indices.numel() > 0:
                    active_samples = samples[new_local_alive_indices]  
                    last_states = active_samples[:, -1, :state_dim]    
                    normalized_states = normalize_fn(torch.cat([
                        last_states,
                        torch.zeros((last_states.size(0), action_dim), device=device)
                    ], dim=1))[:, :state_dim]
                    conditions = {0: normalized_states}
                
                total_generated += steps_added
            
            print(f"Generated {total_generated} steps")

            return all_trajectories, end_indices
    
        # Generate trajectories
        all_samples, end_indices = generate_full_trajectory(
            diffusion_model=self.diffusion_model,
            normalize_fn=self.normalizer,
            unnormalize_fn=self.unnormalizer,
            is_terminated_fn=is_terminated,
            initial_state=self.get_initial_state(env_name=self.env_name ,device=self.device),
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=self.device,
            batch_size=self.num_samples,
            trajectory_max_length=self.T_gen,
            mini_trajectory_size = self.T
        )
        samples = all_samples.detach().cpu().numpy()
        end_indices = end_indices.cpu().numpy()
        
        if self.save:
            os.makedirs('samples', exist_ok=True)
            np.save('samples/diffusion_samples.npy', samples[:5])
            np.save('samples/end_indices.npy', end_indices[:5])

        all_rewards = []

        for i in range(samples.shape[0]):
            sum_reward = 0
            gamma_t = 1
            T_i = end_indices[i] - 1  # Actual length of trajectory i
            for t in range(T_i):
                state = samples[i, t, :self.state_dim]
                action = samples[i, t, self.state_dim:]

                # Compute reward
                if self.reward_fn is not None:
                    reward = self.reward_fn(self.env, state, action)
                else:
                    reward = reward_estimator.predict(np.concatenate([state, action]).reshape(1, -1)).mean()
                sum_reward += reward * gamma_t
                gamma_t *= gamma

            all_rewards.append(sum_reward)

        # Statistics
        mean_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards)
        print(f"Mean reward: {mean_reward}, Std: {std_reward}")

        # Optional: Filter rewards if necessary
        # For example, remove outliers or very low rewards
        threshold = -10000
        filtered_rewards = [r for r in all_rewards if r > threshold]
        print(filtered_rewards)
        print(end_indices)
        print(f"Kept {len(filtered_rewards)} out of {len(all_rewards)} trajectories after filtering")
        print(f"Filtered Mean reward: {np.mean(filtered_rewards)}, Std: {np.std(filtered_rewards)}")

        return np.mean(filtered_rewards)