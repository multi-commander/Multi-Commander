from gym.envs.registration import register

register(
    id='cityflow-v0',
    entry_point='gym_cityflow.envs:CityflowGymEnv',
)