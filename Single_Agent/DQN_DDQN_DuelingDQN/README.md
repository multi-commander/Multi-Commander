# Multi-Commander
Multi-agent signal control

Implementation of DQN, Double DQN and Dueling DQN

### usage
**Training**

*DQN*
```
python run_rl_control.py --algo DQN --epoch 10000 --num_step 2000
```
*Double DQN*
```
python run_rl_control.py --algo DDQN --epoch 101 --num_step 2000
```
*Dueling DQN*
```
python run_rl_control.py --algo DuelDQN --epoch 10
```

**Inference**

*DQN*
```
python run_rl_control.py --algo DQN --inference --num_step 950 --ckpt model/DQN_20190731_144939/DQN-200.h5
```
*DDQN*
```
python run_rl_control.py --algo DDQN --inference --num_step 2500 --ckpt model/DDQN_20190801_085209/DDQN-100.h5
```
*Dueling DQN*
```
python run_rl_control.py --algo DuelDQN --inference --ckpt model/DuelDQN_20190730_165409/DuelDQN-ckpt-10


**Simulation**
```
. simulation.sh

open firefox with the url: http://localhost:8080/?roadnetFile=roadnet.json&logFile=replay.txt
```


