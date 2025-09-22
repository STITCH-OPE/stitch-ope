import json
import numpy as np
import torch as th
import jax.numpy as jnp
from scipy.stats import spearmanr
import os

import optax

from opelab.core.agent import DQN
from opelab.core.baselines.simple import OnPolicy
from opelab.core.baseline import Baseline
from opelab.core.data import Data
from opelab.core.policy import Policy, EpsilonGreedy, UniformContinuous, MixturePolicy

def run_all(env, target:Policy, behavior:Policy, baselines:dict,
            horizon:int, rollouts:int, gamma:float=1.0, trials:int=10,
            oracle_rollouts:int=1000, save_path=None) -> None:
    
    # evaluate the oracle
    online_data = Data(env).make(target, horizon=horizon, rollouts=oracle_rollouts)
    correct = OnPolicy().evaluate(online_data, gamma=gamma)
    
    # evaluate each baseline and compute MSE
    errors = {name: [] for name in baselines}
    values = {name: [] for name in baselines}
    for run in range(trials):
        print('\n' + f'starting trial {run}...')        
        offline_data = Data(env).make(behavior, horizon=horizon, rollouts=rollouts) 
        #reward_estimator = Data(env).train_reward_estimator(offline_data)
        reward_estimator = None

        for name, baseline in baselines.items():
            estimate = baseline.evaluate(offline_data, target, behavior, gamma=gamma, reward_estimator=reward_estimator)
            estimate = np.asarray(estimate).item()
            errors[name].append((estimate - correct) ** 2)
            values[name].append(estimate)
            print(f'[{run}] {name}: estimated {estimate}, correct {correct}')       
    log_rmses = {name: np.log(np.sqrt(np.mean(errors_i))) 
                 for name, errors_i in errors.items()}
    means = {name: np.mean(values_i) for name, values_i in values.items()}
    std = {name: np.std(values_i) for name, values_i in values.items()}
    total_dict = {'log_rmse': log_rmses, 'est': values, 'mean': means, 'std': std, 'correct': correct}
    print(total_dict)
    
    # save results
    if save_path is not None:
        file_name = f'log_rmses_{horizon}_{rollouts}_{gamma}.json'
        with open(os.path.join(save_path, file_name), 'w') as file:
            json.dump(str(total_dict), file, indent=4)


def build_continuous_policies(fully_trained, action_min, action_max, 
                              behavior_noise_prob=0.3, target_noise_prob=0.1):
    behavior_noise = UniformContinuous(action_min, action_max)
    target_noise = UniformContinuous(action_min, action_max)
    behavior_policy = MixturePolicy([behavior_noise, fully_trained], 
                                    [behavior_noise_prob, 1.0 - behavior_noise_prob])
    target_policy = MixturePolicy([target_noise, fully_trained], 
                                  [target_noise_prob, 1.0 - target_noise_prob])
    return behavior_policy, target_policy
    

def train_behavior_policy_dqn(env_factory, layers, 
                              lr=0.001, gamma=0.99, epsilon=0.1, rollouts=500, 
                              rollout_length=500, test_rollouts=5, save_path=None): 
    
    # train agent
    env = env_factory(None)
    state, _ = env.reset()
    agent = DQN(layers, state, opt=optax.adam(lr), gamma=gamma)
    policy = EpsilonGreedy(agent, epsilon)
    for k in range(rollouts):
        total_reward, total_loss = 0.0, 0.0
        state, _ = env.reset()
        for steps in range(rollout_length):
            action = policy.sample(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_loss += agent.update(state, action, reward, next_state, terminated)
            state = next_state
            total_reward += reward
            if terminated:
                break
        print(f'Episode {k} steps {steps} reward {total_reward} loss {total_loss}')
    
    # test and visualize agent
    env = env_factory('human')
    policy = EpsilonGreedy(agent, 0.01)
    for k in range(test_rollouts):
        total_reward = 0.0
        state, _ = env.reset()
        for steps in range(rollout_length):
            action = policy.sample(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated:
                break
        print(f'Episode {k} steps {steps} test reward {total_reward}')
    
    # save agent
    agent.save(save_path)        
        
    return agent

class Transform:
    
    def __init__(self):
        pass
    
    def forward(self, x):
        raise NotImplementedError
    
    def inverse(self, y):
        raise NotImplementedError

class TanhBijector(Transform):
    """
    Bijective transformation of a probability distribution
    using a squashing function (tanh)

    :param epsilon: small value to avoid NaN due to numerical imprecision.
    """

    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    @staticmethod
    def forward(x: th.Tensor) -> th.Tensor:
        return th.tanh(x)

    @staticmethod
    def atanh(x: th.Tensor) -> th.Tensor:
        """
        Inverse of Tanh

        Taken from Pyro: https://github.com/pyro-ppl/pyro
        0.5 * torch.log((1 + x ) / (1 - x))
        """
        return 0.5 * (x.log1p() - (-x).log1p())

    @staticmethod
    def inverse(y: th.Tensor) -> th.Tensor:
        """
        Inverse tanh.

        :param y:
        :return:
        """
        eps = th.finfo(y.dtype).eps
        # Clip the action to avoid NaN
        return TanhBijector.atanh(y.clamp(min=-1.0 + eps, max=1.0 - eps))

    def log_prob_correction(self, x: th.Tensor) -> th.Tensor:
        # Squash correction (from original SAC implementation)
        return th.log(1.0 - th.tanh(x) ** 2 + self.epsilon)
    
    
def create_baselines(env, target_policies, behavior_policy, baseline_configs):
    """
    Dynamically create baselines for evaluation with target policies.
    """
    baselines = {}
    import inspect
    for name, config in baseline_configs.items():
        baselines[name] = []
        if name != "Diffuser":
            baselines[name].append(config["class"](
                **config["params"]
            ))
        for idx, target_policy in enumerate(target_policies):
            if name == "Diffuser":
                baselines[name].append(config["class"](
                    **config["params"],
                    target_model=target_policy,
                    behavior_model=behavior_policy,
                    env=env
                ))
            else:
                baselines[name].append(baselines[name][0])
    return baselines


def evaluate_policies(
    env, 
    target_policies: list, 
    behavior_policy: Policy, 
    baselines: dict, 
    terminate_fn, 
    horizon: int, 
    rollouts: int, 
    gamma: float = 1.0, 
    trials: int = 10, 
    oracle_rollouts: int = 1000, 
    save_path: str = None, 
    top_k: int = 5,
    d4rl: bool = True,
    dataset_path: str = None
):
    """
    Evaluate policies using regret@k, Spearman correlation, normalized returns, and estimate statistics.
    """
    correct_values = []
    for target in target_policies:
        online_data = Data(env).make(target, horizon=horizon, rollouts=oracle_rollouts)
        value = np.mean(OnPolicy().evaluate(online_data, gamma=gamma))
        correct_values.append(value)

    print(f"Correct Values: {correct_values}")
    
    V_min = np.min(correct_values)
    V_max = np.max(correct_values)
    normalized_correct_values = (correct_values - V_min) / (V_max - V_min)

    print(f"Normalized Correct Values: {normalized_correct_values}")

    errors = {name: [] for name in baselines}
    spearman_scores = {name: [] for name in baselines}
    regrets = {name: [] for name in baselines}
    regrets_at_1 = {name: [] for name in baselines}  # Regret@1
    normalized_estimates = {name: {i: [] for i in range(len(target_policies))} for name in baselines}  # Store normalized estimates per target policy
    raw_estimates = {name: {i: [] for i in range(len(target_policies))} for name in baselines}  # Store raw estimates
    raw_correct_values = correct_values  # Save raw correct values as they are

    reward_estimator = None
    if d4rl:
        offline_data = Data(env).load_d4rl_dataset()
        print("Started Training Reward Estimator...")
        reward_estimator = Data(env).train_reward_estimator(offline_data)
    else:
        assert dataset_path is not None, "Please provide a dataset path for offline evaluation."
        offline_data = Data(env).load_dataset(dataset_path)
        print("Loaded Offline Dataset...")
        print("Started Training Reward Estimator...")
        reward_estimator = Data(env).train_reward_estimator(offline_data)
        
    for baselines_name, baseline_instances in baselines.items():
        for baseline_instance in baseline_instances:
            baseline_instance.load_data(offline_data)

    for run in range(trials):
        env.seed(run)
        np.random.seed(run)
        th.manual_seed(run)
                
        
        print(f"\nStarting trial {run + 1}/{trials}...")
        
        for name, baseline_instances in baselines.items():
            trial_estimates = []
            trial_raw_estimates = []  # Store raw estimates for this trial
            for idx, target_policy in enumerate(target_policies):
                estimate = baseline_instances[idx].evaluate(
                    offline_data, target_policy, behavior_policy, 
                    gamma=gamma, reward_estimator=reward_estimator
                )
                mean_estimate = np.mean(estimate)
                normalized_est = (mean_estimate - V_min) / (V_max - V_min)  # Normalize
                normalized_estimates[name][idx].append(normalized_est)  # Store normalized estimate for this trial
                raw_estimates[name][idx].append(mean_estimate)  # Store raw estimate for this trial
                trial_estimates.append(normalized_est)
                trial_raw_estimates.append(mean_estimate)

            mse = np.mean((np.array(trial_estimates) - normalized_correct_values) ** 2)
            errors[name].append(mse)

            spearman_corr, _ = spearmanr(normalized_correct_values, trial_estimates)
            spearman_scores[name].append(spearman_corr)

            top_k_indices = np.argsort(-np.array(trial_estimates))[:top_k]
            best_in_top_k = np.max(normalized_correct_values[top_k_indices])
            regret = np.max(normalized_correct_values) - best_in_top_k
            regrets[name].append(regret)

            # Compute regret@1
            best_in_top_1 = np.max(normalized_correct_values[np.argsort(-np.array(trial_estimates))[:1]])
            regret_at_1 = np.max(normalized_correct_values) - best_in_top_1
            regrets_at_1[name].append(regret_at_1)

            print(f"[Trial {run + 1}] {name}: MSE={mse:.4f}, Spearman={spearman_corr:.4f}, Regret@{top_k}={regret:.4f}, Regret@1={regret_at_1:.4f}")

    # Compute mean and std across trials for each target policy
    mean_estimates = {name: [] for name in baselines}
    std_estimates = {name: [] for name in baselines}
    
    for name in baselines:
        for idx in range(len(target_policies)):
            mean_estimates[name].append(np.mean(normalized_estimates[name][idx]))
            std_estimates[name].append(np.std(normalized_estimates[name][idx]))

    # Aggregate results
    results = {
        "log_rmse": {name: np.log(np.sqrt(np.mean(errors[name]))) for name in baselines},
        "mean_spearman": {name: np.mean(spearman_scores[name]) for name in baselines},
        "mean_regret": {name: np.mean(regrets[name]) for name in baselines},
        "mean_regret_at_1": {name: np.mean(regrets_at_1[name]) for name in baselines},  # Regret@1
        "mean_mse": {name: np.mean(errors[name]) for name in baselines},
        "normalized_correct_values": normalized_correct_values.tolist(),
        "raw_correct_values": raw_correct_values,  # Raw correct values
        "normalized_estimates": {name: {i: normalized_estimates[name][i] for i in range(len(target_policies))} for name in baselines},
        "raw_estimates": {name: {i: raw_estimates[name][i] for i in range(len(target_policies))} for name in baselines},  # Raw estimates
        "mean_estimates": {name: mean_estimates[name] for name in baselines},
        "std_estimates": {name: std_estimates[name] for name in baselines},
    }

    print("Experiment Results:", results)

    def json_default(obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert numpy arrays to lists
        elif isinstance(obj, jnp.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


    if save_path:
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, "experiment_results.json")
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=4, default=json_default)

    return results



def evaluate_method_for_sweep(
    env, 
    target_policies: list, 
    behavior_policy: Policy, 
    baseline: Baseline, 
    terminate_fn, 
    horizon: int, 
    rollouts: int, 
    gamma: float = 1.0, 
    trials: int = 10, 
    oracle_rollouts: int = 1000, 
    save_path: str = None, 
    top_k: int = 5,
    d4rl: bool = True,
    dataset_path: str = None
):
    """
    Evaluate policies using regret@k, Spearman correlation, normalized returns, and estimate statistics.
    """
    correct_values = []
    for target in target_policies:
        online_data = Data(env).make(target, horizon=horizon, rollouts=oracle_rollouts)
        value = np.mean(OnPolicy().evaluate(online_data, gamma=gamma))
        correct_values.append(value)

    print(f"Correct Values: {correct_values}")
    
    V_min = np.min(correct_values)
    V_max = np.max(correct_values)
    normalized_correct_values = (correct_values - V_min) / (V_max - V_min)

    print(f"Normalized Correct Values: {normalized_correct_values}")

    
    reward_estimator = None
    if d4rl:
        offline_data = Data(env).load_d4rl_dataset()
        print("Started Training Reward Estimator...")
        reward_estimator = Data(env).train_reward_estimator(offline_data)
    else:
        assert dataset_path is not None, "Please provide a dataset path for offline evaluation."
        offline_data = Data(env).load_dataset(dataset_path)
        print("Loaded Offline Dataset...")
        print("Started Training Reward Estimator...")
        reward_estimator = Data(env).train_reward_estimator(offline_data)
        
    
    baseline.load_data(offline_data)

    
    env.seed(0)
    np.random.seed(0)
    th.manual_seed(0)
            
    
    print(f"\nStarting trial ")
    
    
    trial_estimates = []
    trial_raw_estimates = []  # Store raw estimates for this trial
    for idx, target_policy in enumerate(target_policies):
        estimate = baseline.evaluate(
            offline_data, target_policy, behavior_policy, 
            gamma=gamma, reward_estimator=reward_estimator
        )
        mean_estimate = np.mean(estimate)
        normalized_est = (mean_estimate - V_min) / (V_max - V_min)  # Normalize
        #normalized_estimates[name][idx].append(normalized_est)  # Store normalized estimate for this trial
        #raw_estimates[name][idx].append(mean_estimate)  # Store raw estimate for this trial
        trial_estimates.append(normalized_est)
        trial_raw_estimates.append(mean_estimate)

    mse = np.mean((np.array(trial_estimates) - normalized_correct_values) ** 2)
    #errors[name].append(mse)

    spearman_corr, _ = spearmanr(normalized_correct_values, trial_estimates)
    #spearman_scores[name].append(spearman_corr)

    top_k_indices = np.argsort(-np.array(trial_estimates))[:top_k]
    best_in_top_k = np.max(normalized_correct_values[top_k_indices])
    regret = np.max(normalized_correct_values) - best_in_top_k
    #regrets[name].append(regret)

    # Compute regret@1
    best_in_top_1 = np.max(normalized_correct_values[np.argsort(-np.array(trial_estimates))[:1]])
    regret_at_1 = np.max(normalized_correct_values) - best_in_top_1
    #regrets_at_1[name].append(regret_at_1)

    print(f" {name}: MSE={mse:.4f}, Spearman={spearman_corr:.4f}, Regret@{top_k}={regret:.4f}, Regret@1={regret_at_1:.4f}")



    return mse, spearman_corr
