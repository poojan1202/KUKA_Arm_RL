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
env = Monitor(env,'monitor_0505_3')

#model = SAC('MlpPolicy',env,verbose=1,device='cuda')
#policy_kwargs = dict(activation_fn=th.nn.ReLU,net_arch=[512,512,256,128])
model = PPO.load('arm_0505_02',env)
#model = PPO('MlpPolicy',env,verbose=1)
model.learn(500000)
t = env.get_episode_rewards()
model.save("arm_0505_03")
del model




file_name = "rewards_0504_06_nm.pkl"
op_file = open(file_name,'wb')
pickle.dump(t, op_file)
op_file.close()

fi,a = plt.subplots()
a.plot(np.arange(len(t)),t)
plt.show()
