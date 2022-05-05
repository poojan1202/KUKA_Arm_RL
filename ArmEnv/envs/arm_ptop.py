## Point to Point
'''... Description ...'''

import gym
import numpy as np

import pybullet as p
from ArmEnv.ents.arm import Arm
from ArmEnv.ents.plane import Plane
from ArmEnv.ents.goal import Goal
#from univ_arm.resources.goal import Goal

class PointToPoint(gym.Env):
    metadata = {'render_modes':['human']}

    def __init__(self,gui=False,mode='T',record=False):
        self.mode = mode
        self.record = record


        # the ranges for action spaces was decided based on tinkering done in testing2.py
        if(mode == 'T'):
            self.action_space = gym.spaces.box.Box(
                low = np.array([-200, -200, -200, -200, -200, -200,-200]),
                high = np.array([200,  200,  200,  200,  200,  200, 200])
            )
        elif(mode == 'V'):
            self.action_space = gym.spaces.box.Box(
                low = np.array([-0.01, -0.01, -0.01, -0.01, -0.01, -0.01,-0.01]),
                high = np.array([0.01,  0.01,  0.01,  0.01,  0.01,  0.01, 0.01])
            )
        else:
            self.action_space = gym.spaces.box.Box(
                low = np.array([-5, -5, -5, -5, -5, -5,-5]),
                high = np.array([5,  5,  5,  5,  5,  5, 5])
            )
        self.observation_space = gym.spaces.box.Box(    #change later
            low = np.array([-100,-10, -100,-10, -100,-10, -100,-10, -100,-10, -100,-10, -100,-10]),
            high = np.array([100, 10,  100, 10,  100, 10,  100, 10,  100, 10,  100, 10,  100, 10])
        )


        self.np_random, _ = gym.utils.seeding.np_random()
        if gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        #p.setTimeStep(1/30, self.client)


        ## SUBJECT TO CHANGE
        self.timesteps = 0
        #original__self.max_timesteps = 5000
        self.max_timesteps = 5000
        self.arm = None
        #self.goal = None
        #self.goal_box = None #named as such becoz the random coordinates are named goal here
        self.done = False
        #self.prev_dist_to_goal = None
        self.rendered_image = None
        self.render_rot_matrix = None
        self.distance_from_gripper = 0
        self.logging_id = 0
        self.reset()

        

    def step(self,action):
        
        self.timesteps += 1
        for i in range(4):
            self.arm.apply_action(action,self.mode)
            p.stepSimulation()
        arm_ob = self.arm.get_observation()
        reward = 0  #initialize reward 0

        #pos,_= p.getBasePositionAndOrientation(self.goal_box.box)
        
        eeloc = p.getLinkState(self.arm.arm,6)[0] # 6th link is end effector (probably)

        reward = -1*np.linalg.norm(np.array(eeloc)-np.array([0.5,0,0.5]))

        if self.timesteps > self.max_timesteps:
            self.done = True

            
        return arm_ob, reward, self.done, dict()


    def reset(self):
        self.timesteps = 0
        p.resetSimulation(self.client)
        p.setGravity(0,0,-10)
        # Reload Plane and Car
        self.arm = Arm(self.client)
        Plane(self.client)
        Goal(self.client,[0.5,0,0.5])


        # Set Random Goal   #this will not be used as position of goal is hardcoded in goal.py
        x = (self.np_random.uniform(5, 9) if self.np_random.randint(2) else self.np_random.uniform(-5,-9))
        y = (self.np_random.uniform(5, 9) if self.np_random.randint(2) else self.np_random.uniform(-5,-9))
        self.goal = (x,y)
        self.done = False

        #self.goal_box = Goal(self.client, self.goal)

        arm_ob = self.arm.get_observation()

        #self.prev_dist_to_goal = math.sqrt(((car_ob[0] - self.goal[0])**2 + (car_ob[1] - self.goal[1])**2 ))
        
        #return np.array(car_ob + self.goal, dtype=np.float32)
        if(self.record):
            self.logging_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4,fileName='rec.mp4')
        return arm_ob

    def render(self):
        pass

    def close(self):
        if(self.record):
            p.stopStateLogging(self.logging_id)
        p.disconnect(self.client)

    def seed(self,seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
