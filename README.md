# STITCH-OPE

<p align="center">
    <a href= "https://arxiv.org/abs/2505.20781">
        <img src="https://img.shields.io/badge/arXiv-2505.20781-b31b1b" /></a>
</p>

<p align="center">
    <img src="media/trailer.gif" alt="animated" width="75%"/>
</p>

This repository contains the code for the paper [STITCH-OPE: Trajectory Stitching with Guided Diffusion for Off-Policy Evaluation](https://stitch-ope.github.io/).

## Setting up the environment
There are two ways to set up the environment, using our dockerfile with the provided Makefile (This path is much easier), or setting up a conda environment.
### 1- Docker (Recommended)
```
sudo make build      # build the GPU image
sudo make sanity     # quick check: imports mujoco_py/gym/d4rl + steps Hopper to ensure mujoco works
```
to automatically download the assets (policies, diffusion models, datasets, etc) run: 
```
# Pick ONE of these; assets are saved into your repo on the host:
# since the docker mounts your repo at /workspace, you can also download it and put inside the host (In case google Drive link is not accessible)
sudo make download ZIP_LOCAL=assets.zip
sudo make download ZIP_URL="https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing"

```
After these you have full access to the running code and a shell:
```
sudo make shell
```
Please read the other sections regarding how to run the experiments and train models, you are good to go!

### 2- Setup a conda environment on python 3.9 (Not recommended)
```bash
conda create -n ope python=3.9
conda activate ope
```
### Installing D4RL
You need to install D4RL in order to run the experiments. You can do this by running the following command:
```bash
pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
```

To use MuJoCo 2.1.0, you need a license key and the correct binaries.
please follow the instructions on the [OpenAI mujoco website](hhttps://github.com/openai/mujoco-py) to get the license key and install the correct binaries.

Once D4RL is installed, you can safely update the environment using the provided env.yml:

```bash
conda env update -n ope -f env.yml
```

If you want to also run the diffusion policy experiments, you need to install clean diffuser library. Instructions can be found [here](https://cleandiffuserteam.github.io/CleanDiffuserDocs/docs/introduction/installation/installation.html).


## Working with OPE-Lab

This framework supports running off-policy evaluation (OPE) experiments on two types of environments:
1. D4RL benchmark environments (with pre-collected datasets)
2. Standard Gym environments (with custom dataset generation)

### Dataset Management

#### D4RL Environments
For D4RL environments, the datasets are already included in the D4RL package, so no additional dataset generation is required.

#### Gym Environments
For Gym environments, you need to generate datasets using the provided script:

```bash
python opelab/examples/gym/generate_dataset.py --name dataset_name
```

This script:
- Collects trajectories using a specified policy
- Stores observations, actions, rewards, and terminal states
- Computes and saves normalization statistics for the dataset
- Saves everything to the `dataset/dataset_name/` directory

### Training Diffusion Models

For both environment types, you can train diffusion models:

#### For D4RL Environments:
```bash
python opelab/examples/d4rl/diffusion_trainer.py --dataset hopper-medium-v2 --T 16 --D 256 --epochs 100 --output_dir ./trained_models
```

#### For Gym Environments:
```bash
python opelab/examples/gym/diffusion_trainer.py --env Pendulum-v1 --T 2 --D 128 --epochs 100 --train_steps 100 --batch_size 64 --output_dir ./trained_models
```

### Configuring Experiments

Create JSON configuration files to define the experiment setup. Separate configurations are needed for D4RL and Gym environments.

#### D4RL Configuration Example:
```json
{
    "env_name": "hopper-medium-v2",
    "guidance_hyperparams": {
        "action_scale": 0.5,
        "normalize_grad": true,
        "k_guide": 1,
        "use_neg_grad": true,
        "ratio": 0.5
    },
    "target_policy_paths": [
        "policy/hopper/dope/1.pkl",
        "policy/hopper/dope/2.pkl"
    ],
    "baseline_configs": {
        "Naive": {
            "class": "OnPolicy",
            "params": {}
        },
        "Diffuser": {
            "class": "Diffuser",
            "params": {
                "T": 16,
                "D": 256,
                "num_samples": 50,
                "model_path": "path/to/diffusion/model.pth"
            }
        }
    },
    "experiment_params": {
        "horizon": 768,
        "rollouts": 50,
        "gamma": 0.99,
        "trials": 5,
        "save_path": "results/experiment_name"
    }
}
```

#### Gym Configuration Example:
```json
{
    "env_name": "Pendulum-v1",
    "behavior_policy_path": "policy/pendulum/Pi_3.pkl",
    "dataset_path": "dataset/pdataset/",
    "guidance_hyperparams": {
        "action_scale": 0.1,
        "normalize_grad": true,
        "k_guide": 1
    },
    "target_policy_paths": [
        "policy/pendulum/Pi_1.pkl",
        "policy/pendulum/Pi_2.pkl"
    ],
    "baseline_configs": {
        "Naive": {
            "class": "OnPolicy",
            "params": {}
        },
        "Diffuser": {
            "class": "Diffuser",
            "params": {
                "T": 2,
                "D": 256,
                "num_samples": 100,
                "model_path": "pendulum/T2D256/m1.pth"
            }
        }
    },
    "experiment_params": {
        "horizon": 200,
        "rollouts": 50,
        "gamma": 0.99,
        "trials": 5,
        "save_path": "results/pendulum/experiment_name"
    }
}
```

### Running Experiments with main_full.py

The main entry point for running OPE experiments is `main_full.py`, which is available for both D4RL and Gym environments.

#### For D4RL Environments:
```bash
python opelab/examples/d4rl/main_full.py --config opelab/examples/d4rl/configs/hopper.json
```

#### For Gym Environments:
```bash
python opelab/examples/gym/main_full.py --config opelab/examples/gym/configs/pendulum.json
```

### Experiment Parameters

- `horizon`: Number of timesteps to evaluate
- `rollouts`: Number of environment rollouts to perform
- `gamma`: Discount factor
- `trials`: Number of evaluation trials
- `save_path`: Path to save the experiment results
- `top_k`: Number of top policies to identify
- `oracle_rollouts`: Number of rollouts for the oracle evaluation

 Full model checkpoint and policies is hosted at this  link: [Google Drive](https://drive.google.com/drive/folders/1v3Vi6yaGHXq63MmYrbfU13MCLli2OFfp?usp=sharing).



### Acknowledgements

Our codebase builds upon several open-source projects. We would like to acknowledge the following repositories:

- [Diffuser](https://github.com/jannerm/diffuser)
- [DDPM-Pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)


### Reference

If you find this code useful in your research, please consider citing the following paper:

```@InProceedings{stitch_ope,
  title={STITCH-OPE: Trajectory Stitching with Guided Diffusion for Off-Policy Evaluation},
  author={Hossein Goli and Michael Gimelfarb and Nathan Samuel de Lara and Haruki Nishimura and Masha Itkina and Florian Shkurti},
  year={2025},
  booktitle={Neural Information Processing Systems (NeurIPS)}} 
```

