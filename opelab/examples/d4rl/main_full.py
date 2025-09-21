import argparse
import os
import json
import gym
import torch
import numpy as np
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../../..'))

from opelab.core.baselines.simple import OnPolicy
from opelab.core.baselines.diffuser import Diffuser
from opelab.core.policy import D4RLPolicy, D4RLSACPolicy
from opelab.examples.helpers import evaluate_policies, create_baselines

BASELINE_CLASSES = {
        "Diffuser": Diffuser,
        "OnPolicy": OnPolicy,
    }

def main(config_path, device):
    with open(config_path, 'r') as f:
        config = json.load(f)

    env_name = config["env_name"]
    guidance_hyperparams = config["guidance_hyperparams"]
    target_policy_paths = config["target_policy_paths"]
    baseline_configs = config["baseline_configs"]
    experiment_params = config["experiment_params"]

    env = gym.make(env_name)

    env_min = torch.tensor(env.action_space.low, dtype=torch.float32, device=device)
    env_max = torch.tensor(env.action_space.high, dtype=torch.float32, device=device)
    action_bounds = [env_min, env_max]

    behavior_policy = D4RLPolicy(env_name).to(device)

    target_policies = [
        D4RLSACPolicy(path).to(device) for path in target_policy_paths
    ]
    
    reward_fn, terminate_fn = get_environment_specific_functions(env_name)

    def compute_normalization(env_name):
        dataset = gym.make(env_name).get_dataset()
        observations = dataset['observations']
        actions = dataset['actions']
        mean_state, std_state = np.mean(observations, axis=0), np.std(observations, axis=0)
        mean_action, std_action = np.mean(actions, axis=0), np.std(actions, axis=0)
        mean = np.concatenate((mean_state, mean_action))
        std = np.concatenate((std_state, std_action))
        return mean, std

    mean, std = compute_normalization(env_name)
    mean = torch.tensor(mean).to(device)
    std = torch.tensor(std).to(device)
    normalize_fn = lambda x: (x - mean) / std
    unnormalize_fn = lambda x: x * std + mean

    for baseline_name, config in baseline_configs.items():
        if config["class"] in BASELINE_CLASSES:
            config["class"] = BASELINE_CLASSES[config["class"]]
        else:
            raise ValueError(f"Unknown baseline class: {config['class']}, Please add it the core/baselines")
        
    if "Diffuser" in baseline_configs:
        baseline_configs["Diffuser"]["params"].update({
            "action_dim": env.action_space.shape[0],
            "state_dim": env.observation_space.shape[0],
            "device": device,
            "normalizer": normalize_fn,
            "unnormalizer": unnormalize_fn,
            "reward_fn": reward_fn,
            "is_terminated_fn": terminate_fn,
            "guidance_hyperparams": guidance_hyperparams,
        })

    if "PolicyGuidedDiffusion" in baseline_configs:
        baseline_configs["PolicyGuidedDiffusion"]["params"].update({
            "device": device,
            "dataset_name": env_name
        })

    if "MBR" in baseline_configs:
        baseline_configs["MBR"]["params"].update({
            "terminated_fn": terminate_fn,
        })
        print(baseline_configs["MBR"]["params"])

    baselines = create_baselines(env, target_policies, behavior_policy, baseline_configs)

    evaluate_policies(
        env=env,
        target_policies=target_policies,
        behavior_policy=behavior_policy,
        baselines=baselines,
        terminate_fn=terminate_fn,
        **experiment_params
    )

#this is for testing and the reward estimator will be trained and used nevertheless (helpers/create_baseline function)
def get_environment_specific_functions(env_name):
    """
    Returns environment-specific reward_fn and terminate_fn.
    """
    if "hopper" in env_name.lower():
        reward_fn = None
        
        def terminate_fn(state):
            state_np = state.cpu().numpy() if isinstance(state, torch.Tensor) else state
            height = state_np[0]
            ang = state_np[1]
            return not (
                np.isfinite(state_np).all()
                and (np.abs(state_np[2:]) < 100).all()
                and (height > 0.7)
                and (abs(ang) < 0.2)
            )
        return reward_fn, terminate_fn
    
    elif 'cheetah' in env_name.lower():
        reward_fn = None
        
        def terminate_fn(state):
            return False
        
        return reward_fn, terminate_fn

    elif "walker2d" in env_name.lower():
        reward_fn = None
        
        def terminate_fn(state):
            state_np = state.cpu().numpy() if isinstance(state, torch.Tensor) else state
            height = state_np[0] 
            ang = state_np[1]  
            return not (
                np.isfinite(state_np).all()
                and (height > 0.8)  
                and (height < 2.0)  
                and (abs(ang) < 1.0)  
            )
        
        return reward_fn, terminate_fn

    else:
        raise NotImplementedError(f"Environment {env_name} not supported.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the experiment on")
    args = parser.parse_args()
    main(args.config, args.device)
