import ray
ray.init()

import gym
import numpy as np
import cityflow


@ray.remote
class GymEnvironment(object):
	def __init__(self, name):
		self.env = gym.make(name)
		self.env.reset()

	def step(self, action):
		return self.env.step(action)

	def reset(self):
		self.env.reset()

@ray.remote
class CityFlowEnv(object):
	def __init__(self, config_path):
		self.eng = cityflow.Engine(config_path, thread_num=1)

	def step(self):
		return self.eng.next_step()

# def main():
# 	import time
# 	import matplotlib.pyplot as plt

# 	pong = GymEnvironment.remote('Pong-v0') # this is an actor
# 	plt.ion()
# 	while True:
# 		action = np.random.randint(0, 6)
# 		res = ray.get( pong.step.remote(action) )
# 		# print(res)
# 		plt.imshow(res[0])
# 		# time.sleep(0.1)
# 		plt.pause(0.1)

def main():
	config_path = "./examples/config_3x3.json"
	cityEnv = CityFlowEnv.remote(config_path)
	print(config_path)
	for i in range(1000):
		cityEnv.step.remote()

if __name__ == '__main__':
	main()
