## Introduction
This is the code for reproducing the results of the paper \[Hypercube Policy Regularization Framework for Offline Reinforcement Learning]

## Prerequisites
Python 3.7 

mujoco 2.1.0

d4rl[https://github.com/Farama-Foundation/D4RL]

## Installation and Usage

Here is an example of how to install all the dependencies on Ubuntu:
```bash
conda create -n GPC-SAC python=3.7
conda activate GPC-SAC
cd GPC-SAC-master
pip install -r requirements.txt
git clone [https://github.com/Farama-Foundation/D4RL.git]
cd d4rl
# Remove lines including 'dm_control' in setup.py
pip install -e .
```

## Reproducing the results

For running TD3-BC-C on Gym and Adroit environments, run:
```bash
python -m scripts.gpc_sac --env_name [ENVIRONMENT] --seed [K]  --state_n [N]  --alpha [M]
```
For example, to reproduce the GPC-SAC results for halfcheetah-medium-v2, run:
```bash
python main.py --env_name halfcheetah-medium-v2 --seed 0 --state_n 5 --alpha 40.0
```

For running TD3-BC-C on Maze environments, run:
```bash
python -m scripts.gpc_sac --env_name [ENVIRONMENT] --seed [K]  --state_n [N]  --alpha [M] --scale [I] --shift [L]
```
For example to reproduce the TD3-BC-C results for antmaze-medium-diverse-v0, run:
```bash
python main.py --env_name antmaze-medium-diverse-v0 --seed 0 --state_n 5 --alpha 10.0  --scale=100  --shift=-1
```

For running Diffusion-QL-C on different environments, run:
```.bash
python main.py --env_name [ENVIRONMENT] --device 0 --ms online --lr_decay --seed [K]  --state_n [N] 
```

For example to reproduce the Diffusion-QL-C results for walker2d-medium-expert-v2, run:
```.bash
python main.py --env_name walker2d-medium-expert-v2 --device 0 --ms online --lr_decay --state_n 5
```

## Implementation

The core implementation is given in `Hypercube\TD3-BC-C\TD3_BC` and `Hypercube\Diffusion-QL-C\agents\ql_diffusion`, some other changes are in `Hypercube\TD3-BC-C\utils`, `Hypercube\TD3-BC-C\main`, `Hypercube\Diffusion-QL-C\utils\data_sampler` and  `Hypercube\Diffusion-QL-C\main`.

In case of any questions, bugs, suggestions or improvements, please feel free to open an issue.
