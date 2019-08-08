# presslight
[Hua Wei, et al, 2019](http://delivery.acm.org/10.1145/3340000/3330949/p1290-wei.pdf?ip=96.126.102.110&id=3330949&acc=OPEN&key=4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E6D218144511F3437&__acm__=1565080062_f96f49613620b06cf02de457e2abcb3d)
### 解决什么问题

propose a novel RL method for *multi-intersection traffic signal control* on the arterials.

### 使用方法/模型

DQN，每个agent控制自己的Intersection；互相之间无信息传递

### 亮点或者贡献

- draw a connection on the design between reinforcement learning with conventional transportation control methods.
- the first time the individual RL model automatically achieves coordination along arterial without any prior knowledge

### 程序说明
训练过程运行`runexp.py`，对训练好的模型进行`replay.txt`进行生成使用`replay.py`。

Before running above codes, you may need to install following packages or environments:
- Python 3.6
- Keras 2.2.0 
- Tensorflow 1.13.0
