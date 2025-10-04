import argparse
import os
import json
import gym
import torch
import numpy as np
from torch import cos
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../../..'))

from opelab.core.baselines.simple import OnPolicy
from opelab.core.baselines.diffuser import Diffuser
from opelab.core.task import ContinuousAcrobotEnv
from opelab.core.policy import D4RLPolicy, D4RLSACPolicy, TD3Policy
from opelab.examples.helpers import evaluate_policies, create_baselines

BASELINE_CLASSES = {
        "OnPolicy": OnPolicy,
        "Diffuser": Diffuser,
    }

def main(config_path, device):
    with open(config_path, 'r') as f:
        config = json.load(f)

    env_name = config["env_name"]
    guidance_hyperparams = config["guidance_hyperparams"]
    target_policy_paths = config["target_policy_paths"]
    behavior_policy_path = config["behavior_policy_path"]
    dataset_path = config["dataset_path"]
    baseline_configs = config["baseline_configs"]
    experiment_params = config["experiment_params"]
    
    if env_name == "acrobat":
        env = ContinuousAcrobotEnv()
        print(env.spec)
    else:
        env = gym.make(env_name)
        
    env_min = torch.tensor(env.action_space.low, dtype=torch.float32, device=device)
    env_max = torch.tensor(env.action_space.high, dtype=torch.float32, device=device)
    action_bounds = [env_min, env_max]

    behavior_policy = TD3Policy(behavior_policy_path).to(device)
    
    action_bound = env.action_space.high[0]
    print('action_bound:', action_bound)
    target_policies = [
        TD3Policy(path, action_bound=env.action_space.high[0]).to(device) for path in target_policy_paths
    ]
        
    # reward_fn, terminate_fn = get_environment_specific_functions(env_name)

    def compute_normalization(dataset_path):
        with open(os.path.join(dataset_path, "normalization.json"), 'r') as f:
            normalization = json.load(f)
        mean_state = normalization["state_mean"]
        std_state = normalization["state_std"]
        mean_action = normalization["action_mean"]
        std_action = normalization["action_std"]
        mean = np.concatenate([mean_state, mean_action])
        std = np.concatenate([std_state, std_action])
        
        return mean, std
        

    mean, std = compute_normalization(dataset_path)
    mean = torch.tensor(mean, dtype=torch.float32).to(device)
    std = torch.tensor(std, dtype=torch.float32).to(device)
    print(mean.dtype)

    normalize_fn = lambda x: (x - mean) / std
    unnormalize_fn = lambda x: x * std + mean

    for baseline_name, config in baseline_configs.items():
        if config["class"] in BASELINE_CLASSES:
            config["class"] = BASELINE_CLASSES[config["class"]]
        else:
            raise ValueError(f"Unknown baseline class: {config['class']}, Please add it the core/baselines")
    
    terminate_fn = lambda x: False
    reward_fn = None
          
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

    baselines = create_baselines(env, target_policies, behavior_policy, baseline_configs)

    evaluate_policies(
        env=env,
        target_policies=target_policies,
        behavior_policy=behavior_policy,
        baselines=baselines,
        terminate_fn=None,
        d4rl=False,
        dataset_path=dataset_path,
        **experiment_params
    )

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the experiment on")
    args = parser.parse_args()
    main(args.config, args.device)
