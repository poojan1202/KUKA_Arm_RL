import gym
import ArmEnv
import matplotlib.pyplot as plt
import numpy as np
import pickle

from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import SAC



env = gym.make('PointToPoint-v0')


#model = SAC('MlpPolicy',env,verbose=1,device='cuda')
#policy_kwargs = dict(activation_fn=th.nn.ReLU,net_arch=[512,512,256,128])
for i in range(10):
    print('EPISODE', i+1)
    done = False
    env.reset()
    while not done:
        act = env.action_space.sample()
        st,reward,done,info = env.step(act)
        print('reward =', reward)
        print('state = ',len(st))
        print('action = ', len(act))