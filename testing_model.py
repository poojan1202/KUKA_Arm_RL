import gym
import ArmEnv
import matplotlib.pyplot as plt
import numpy as np
import pickle

from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import SAC



env = gym.make('PointToPoint-v0',gui=True)

#model = PPO.load("arm_0505_01.zip", env=env,custom_objects={"learning_rate": 0.0,
#            "lr_schedule": lambda _: 0.0,
#            "clip_range": lambda _: 0.0,})
model = PPO.load("arm_0505_02.zip", env=env)

rew = []

for i in range(1):
    done = False
    obs = env.reset()
    while not done:
        action, _state =model.predict(obs)
        obs,reward,done,_ = env.step(action)
        if i==0:
            rew.append(reward)

t = np.arange(len(rew))

fig,ax = plt.subplots()
ax.plot(t,rew)
plt.show()
