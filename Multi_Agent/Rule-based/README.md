# Multi-Commander
Multi-agent signal control

## usage
### Single agent for single intersection
**Training**

*DQN*
```
python run_rl_control.py --algo DQN --epoch 200 --num_step 2000 --phase_step 1
```
*Double DQN*
```
python run_rl_control.py --algo DDQN --epoch 200 --num_step 2000 --phase_step 1
```
*Dueling DQN*
```
python run_rl_control.py --algo DuelDQN --epoch 200 --num_step 2000 --phase_step 1
```

**Inference**

*DQN*
```
python run_rl_control.py --algo DQN --inference --num_step 3000 --ckpt model/DQN_20190803_150924/DQN-200.h5
```
*DDQN*
```
python run_rl_control.py --algo DDQN --inference --num_step 2000 --ckpt model/DDQN_20190801_085209/DDQN-100.h5
```
*Dueling DQN*
```
python run_rl_control.py --algo DuelDQN --inference --num_step 2000 --ckpt model/DuelDQN_20190730_165409/DuelDQN-ckpt-10
```

**Simulation**
```
. simulation.sh

open firefox with the url: http://localhost:8080/?roadnetFile=roadnet.json&logFile=replay.txt
```


### Multiple intersections signal control

**Training**

*QMIX (based on Ray)*
```
python ray_multi_agent.py
```

*MDQN*
```
python run_rl_multi_control.py --algo MDQN --epoch 1000 --num_step 500 --phase_step 10
```

**Inference**

*MDQN*
```
python run_rl_multi_control.py --algo MDQN --inference --num_step 1500 --phase_step 15 --ckpt model/XXXXXXX/MDQN-1.h5
```


### Rule based
*1\*6 roadnet*

Generate checkpoint
```
python run_rl_multi_control.py --algo MDQN --epoch 1 --num_step 1 --phase_step 15
```

Generate replay file
```
python run_rl_multi_control.py --algo MDQN --inference --num_step 1500 --phase_step 15 --ckpt model/XXXXXXX/MDQN-1.h5
```

Simulation
```
. simulation.sh

open firefox with the url: http://localhost:8080/?roadnetFile=roadnet.json&logFile=replay.txt
```

![demo_1_6](./README.assets/demo_1_6.gif)
