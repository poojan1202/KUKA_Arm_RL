import gym
import ArmEnv
import matplotlib.pyplot as plt

from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import SAC


env = gym.make('PointToPoint-v0')

#model = SAC('MlpPolicy',env,verbose=1,device='cuda')
model = PPO('MlpPolicy',env,verbose=1)
model.learn(100000)
env.close()
env = gym.make('PointToPoint-v0',gui=True)
obs = env.reset()
print('Observation:',obs)
dones = False
rews = []

for i in range(10000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    if(dones):
        obs = env.reset()
    if(i%10 == 0):
        #print(rewards)
        rews.append(rewards)

plt.plot(rews)
plt.savefig('rews.png')
