# Multi-Commander
### Traffic Signal Control based on Multi & Single Agent Reinforcement learning Algorithms
> This is a project for Deecamp 2019 which cooperated with  APEX Lib of Shanghai Jiaotong University and Tianrang Intelligence Company

#### [Documentation for Q-Value based Method](./Single_Agent/DQN_DDQN_DuelingDQN/README.md)

[DQN](./Single_Agent/DQN_DDQN_DuelingDQN/dqn_agent.py)

[DDQN](./Single_Agent/DQN_DDQN_DuelingDQN/dqn_agent.py)

[DuelingDQN](./Single_Agent/DQN_DDQN_DuelingDQN/duelingDQN.py)

[RayDQN](./Single_Agent/RayDQN_Perfect/README.md)



#### Documentation for PG based Method

PPO

DDPG

TD3



#### Documentation for Distributed Method

IMPALA

A3C

APPO

Ape-X



#### Multi-Agent Method

[QMIX](./Multi_Agent/QMIX&Rule-based/README.md)

VDN

[Gamma-Reward](./Multi_Agent/Gamma_reward/README.md)

[PressLight](./Multi_Agent/presslight/README.md)



### Installation

This project is based on [CityFlow](https://cityflow.readthedocs.io/en/latest/) which is *a multi-agent reinforcement learning environment for large scale city traffic scenario*, the algorithms are bases on [Ray](https://ray.readthedocs.io/en/latest/) which is *a fast and simple framework for building and running distributed applications.*

#### [CityFlow build from source](https://cityflow.readthedocs.io/en/latest/install.html)

This guide is based on Ubuntu 16.04.

CityFlow has little dependencies, so building from source is not scary.

1. Check that you have python 3.6 installed. Other version of python might work, however, we only tested on python with version >= 3.6.
2. **Install cpp dependencies**

```
apt update && apt-get install -y build-essential libboost-all-dev cmake
```

3. Clone CityFlow project from github.

```
git clone --recursive https://github.com/cityflow-project/CityFlow.git
```

Notice that CityFlow uses pybind11 to integrate C++ code with python, the repo has pybind11 as a submodule, please use `--recursive` to clone all codes.

4. Go to CityFlow project’s root directory and run

```python
pip install .
```

5. Wait for installation to complete and CityFlow should be successfully installed.

```python
import cityflow
eng = cityflow.Engine
```

#### [Ray Installation](https://ray.readthedocs.io/en/latest/installation.html)

```python
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -U ray  # also recommended: ray[debug]
```

#### [Gym](https://gym.openai.com/docs/)
```
git clone https://github.com/openai/gym
cd gym
pip install -e .
```
