import numpy as np
import torch
import gym
import argparse
import os
import sys
import pickle
import utils
import TD3

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../../../..'))

from opelab.core.task import InfiniteHorizonAcrobotEnv, ContinuousAcrobotEnv

# Runs policy for X episodes and returns average reward
def eval_policy(policy, env_name, seed, eval_episodes=10):
    if env_name == "CAcrobat":
        eval_env = ContinuousAcrobotEnv()
    else:
        eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward

def save_policy_weights(actor, save_file_name):
    weights = {
        'l1/weight': actor.l1.weight.detach().cpu().numpy(),
        'l1/bias': actor.l1.bias.detach().cpu().numpy(),
        'l2/weight': actor.l2.weight.detach().cpu().numpy(),
        'l2/bias': actor.l2.bias.detach().cpu().numpy(),
        'l3/weight': actor.l3.weight.detach().cpu().numpy(),
        'l3/bias': actor.l3.bias.detach().cpu().numpy(),
        'max_action': actor.max_action,
    }
    with open(save_file_name, 'wb') as f:
        pickle.dump(weights, f)
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Hopper-v2")          # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=10e3, type=int)# Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=50e3, type=int)   # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1, type=float)    # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)         # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()

    file_name = f"TD3_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: TD3, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    # Create environment-specific directories
    env_dir = f"./results/{args.env}"
    env_models_dir = f"./models/{args.env}"
    for directory in [env_dir, env_models_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    eval_log_path = f"{env_dir}/{file_name}_evaluation_log.txt"
    eval_results_path = f"{env_dir}/{file_name}"

    #perhaps we shoould register instead of this but it's fine for now
    if args.env == "CAcrobat":
        env = ContinuousAcrobotEnv()
    else:
        env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        "policy_noise": args.policy_noise * max_action,
        "noise_clip": args.noise_clip * max_action,
        "policy_freq": args.policy_freq,
    }

    # Initialize TD3 policy
    policy = TD3.TD3(**kwargs)

    # Load model if specified
    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        load_path = f"{env_models_dir}/{policy_file}.pkl"
        with open(load_path, 'rb') as f:
            policy.actor = pickle.load(f)

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    
    # Evaluate untrained policy
    evaluations = [eval_policy(policy, args.env, args.seed)]
    with open(eval_log_path, "a") as log_file:
        log_file.write(f"Initial Policy Evaluation: {evaluations[-1]:.3f}\n")

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    # Calculate time steps to save policies
    save_interval = int(args.max_timesteps / 10)

    for t in range(int(args.max_timesteps)):
        
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                policy.select_action(np.array(state))
                + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action) 
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done: 
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, args.env, args.seed))
            np.save(eval_results_path, evaluations)

        # Save policy at evenly spaced intervals
        if (t + 1) % save_interval == 0:
            if args.save_model:
                timestep_str = f"{(t + 1) // 1000}k"
                save_file_name = f"{env_models_dir}/{file_name}_t{timestep_str}.pkl"
                save_policy_weights(policy.actor, save_file_name)
                
                with open(eval_log_path, "a") as log_file:
                    log_file.write(f"Saved Policy at Timestep {timestep_str}: Evaluation = {evaluations[-1]:.3f}\n")

    # Final evaluation and saving
    final_evaluation = eval_policy(policy, args.env, args.seed)
    evaluations.append(final_evaluation)
    np.save(eval_results_path, evaluations)

    if args.save_model:
        final_save_path = f"{env_models_dir}/{file_name}_final.pkl"
        with open(final_save_path, "wb") as f:
            pickle.dump(policy.actor, f)
    
    with open(eval_log_path, "a") as log_file:
        log_file.write(f"Final Policy Evaluation: {final_evaluation:.3f}\n")