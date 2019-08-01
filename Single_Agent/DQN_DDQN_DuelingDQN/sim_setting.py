sim_setting_control = {
    "interval": 1.0,  # seconds of each step
    "threadNum": 1,  # this .so is single thread version, this parameter is useless
    "saveReplay": True,  # set to True if your want to replay the traffic in GUI
    "rlTrafficLight": True,  # set to True to control the signal
    "changeLane": False,  # set to False if changing lane is not considered
}

sim_setting_default = {
    "interval": 1.0,            # seconds of each step
    "threadNum": 1,             # this .so is single thread version, this parameter is useless
    "saveReplay": True,         # set to True if your want to replay the traffic in GUI
    "rlTrafficLight": False,    # set to False to control the signal by default
    "changeLane": False,        # set to False if changing lane is not considered
    "plan": [5, 30, 30, 30, 30, 30, 30, 30, 30]
}