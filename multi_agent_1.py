import gym
import ArmEnv
import matplotlib.pyplot as plt
import pybullet as p
import numpy as np
import random
import pickle

from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import SAC

x = random.uniform(0.0,0.5)
y = random.uniform(-0.4,0.4)
z = random.uniform(0.8,1.2)

#env = gym.make('PointToPoint-v0',gui=True,mode='T',record=True, goal = [x,y,z])

if 0<=y<=0.4:
    mdl_slt=1
else:
    mdl_slt=2

for i in range(1):
    x = random.uniform(0.2, 0.5)
    y = random.uniform(-0.4, 0.4)
    z = random.uniform(0.9, 1.2)

    if 0 <= y <= 0.4:
        mdl_slt = 1

    env = gym.make('PointToPoint-v0',gui=True,mode='T',record=True, goal = [x,y,z])
    goal = [x,y,z]
    print('goal:',goal)
    #env.goal_loc(x,y,z)

    if mdl_slt==1:
        model = PPO.load("logs/rl_model_3542980_steps.zip", env=env, custom_objects={"learning_rate": 0.0,
                                                                                     "lr_schedule": lambda _: 0.0,
                                                                                     "clip_range": lambda _: 0.0, })
    else:
        model = PPO.load("logs/rl_model_2177980_steps.zip", env=env, custom_objects={"learning_rate": 0.0,
                                                                                     "lr_schedule": lambda _: 0.0,
                                                                                     "clip_range": lambda _: 0.0, })

#model = PPO.load("trial.zip", env=env)

    rew = []

    done = False
    obs = env.reset()
    while not done:
        action, _state =model.predict(obs)
        #action = np.array([0,0,0,0,0,0,0])
        obs,reward,done,_ = env.step(action)
        #print(reward)
        #print(obs)
        #print(action)
        #print(action)
        if i==0:
            rew.append(reward)

t = np.arange(len(rew))
print(sum(rew))

fig,ax = plt.subplots()
ax.plot(t,rew)
plt.show()
#goal: [0.4,0,1.1]  loc6:  [0.44738302 0.04233105 1.11781087]
#goal: [0.6,0,0.9]  loc6:  [0.5669553  0.07910315 1.04583075]
#goal: [0.3,0,1.2]  loc6:  [0.27454447 0.03467133 1.20632827]