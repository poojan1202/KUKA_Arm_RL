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

    def __init__(self,gui=False,mode='T',record=False,T_sens = 200, V_sens=1, P_sens = 1):
        self.mode = mode
        self.record = record


        # the ranges for action spaces was decided based on tinkering done in testing2.py
        if(mode == 'T'):
            print('TORQUE CONTROL')
            self.action_space = gym.spaces.box.Box(
                low = np.array([-1, -1, -1, -1, -1, -1,-1]),
                high = np.array([1,  1,  1,  1,  1,  1, 1])
            )
        elif(mode == 'V'):
            print("VELOCITY CONTROL")
            self.action_space = gym.spaces.box.Box(
                low = np.array([-1, -1, -1, -1, -1, -1,-1]),
                high = np.array([1,  1,  1,  1,  1,  1, 1])
            )
        elif (mode == 'P'):
            print("POSITION CONTROL")
            self.action_space = gym.spaces.box.Box(
                low=np.array([-1, -1, -1, -1, -1, -1, -1]),
                high=np.array([1, 1, 1, 1, 1, 1, 1])
            )
        else:
            self.action_space = gym.spaces.box.Box(
                low = np.array([-5, -5, -5, -5, -5, -5,-5]),
                high = np.array([5,  5,  5,  5,  5,  5, 5])
            )
        self.observation_space = gym.spaces.box.Box(    #change later
            # low=np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]),
            # high=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
            low = np.array([-3.14,-3.14,-3.14,-3.14,-3.14,-3.14,-3.14,-10,-10,-10,-10,-10,-10]),
            high = np.array([3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 10, 10, 10, 10, 10, 10])
        )


        self.np_random, _ = gym.utils.seeding.np_random()
        if gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        #p.setTimeStep(1/30, self.client)


        ## SUBJECT TO CHANGE
        self.timesteps = 0
        self.max_timesteps = 6500
        self.T_sens = T_sens
        self.V_sens = V_sens
        self.P_sens = P_sens
        self.arm = None
        #self.goal = None
        #self.goal_box = None #named as such becoz the random coordinates are named goal here
        self.done = False
        #self.goal = [0.3,-0.5,1.1]
        self.goal = [0.5,0,0.9]
        #self.prev_dist_to_goal = None
        self.rendered_image = None
        self.render_rot_matrix = None
        self.distance_from_gripper = 0
        self.logging_id = 0
        self.reset()

    def goal_loc(self, x,y,z):
        self.goal[0] = x
        self.goal[1] = y
        self.goal[2] = z
        

    def step(self,action):
        
        self.timesteps += 1
        for i in range(4):          #ADD THIS AS AN ARGUMENT TO ENV CONSTRUCTOR
            self.arm.apply_action(action,self.mode,torque_sens=self.T_sens,vel_sens=self.V_sens,pos_sens=self.P_sens)
            p.stepSimulation()
        arm_ob = self.arm.get_observation()
        reward = 0  #initialize reward 0

        #pos,_= p.getBasePositionAndOrientation(self.goal_box.box)
        
        eeloc = p.getLinkState(self.arm.arm,6)[0] # 6th link is end effector (probably)

        loc6 = np.array(p.getLinkState(self.arm.arm, 6)[0])
        vel6 = np.array(p.getLinkState(self.arm.arm, 6, computeLinkVelocity=1)[6])

        for i in range(3):
            rel = self.goal[i] - loc6[i]
            arm_ob[i+7]=rel
            #arm_ob.append(rel)
        for i in range(3):
            arm_ob[i+10]=vel6[i]
            #arm_ob.append(vel6[i])


        reward = -1*np.linalg.norm(np.array(eeloc)-np.array(self.goal))
        rel = np.array(eeloc)-np.array(self.goal)
        if abs(rel[0])<0.05 and abs(rel[1])<0.05 and abs(rel[2])<0.05:
            reward=0

        if self.timesteps > self.max_timesteps:
            self.done = True
            print('loc6:', loc6)

            
        return arm_ob, reward, self.done, dict()


    def reset(self):
        self.timesteps = 0
        p.resetSimulation(self.client)
        p.setGravity(0,0,-10)
        # Reload Plane and Car
        self.arm = Arm(self.client,)
        Plane(self.client)
        Goal(self.client,self.goal)


        # Set Random Goal   #this will not be used as position of goal is hardcoded in goal.py
        x = (self.np_random.uniform(5, 9) if self.np_random.randint(2) else self.np_random.uniform(-5,-9))
        y = (self.np_random.uniform(5, 9) if self.np_random.randint(2) else self.np_random.uniform(-5,-9))
        #self.goal = (x,y)
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
