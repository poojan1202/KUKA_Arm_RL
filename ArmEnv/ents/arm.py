import pybullet as p


class Arm:
    def __init__(self,client):
        self.client = client

        f_name = 'ArmEnv/resources/model.urdf'
        
        self.arm = p.loadURDF(fileName = f_name,basePosition = [0,0,0.005],useFixedBase=1,physicsClientId = client)
        #p.setJointMotorControlArray(self.arm, [2,4,5],p.POSITION_CONTROL,targetPositions = [-1.2,-1.5,-1.5]) 
        """ for i in range(100):
            p.stepSimulation()

        self.arm_joints = [1,2,3,4] #5 should be 90deg turned
        self.gripper_joints = [9,11,12,14] """

        self.joints = [i for i in range(p.getNumJoints(self.arm))]


    def get_ids(self):
        return self.client, self.id

    def apply_action(self, action, mode):
        ## CHANGE
        ## Make mode changeable
        
        if(mode == 'T'):
            mode = p.TORQUE_CONTROL
            p.setJointMotorControlArray(self.arm,self.joints,mode,forces = action,physicsClientId = self.client)
        elif(mode == 'V'):
            mode = p.VELOCITY_CONTROL
            p.setJointMotorControlArray(self.arm,self.joints,mode,targetVelocities = action,physicsClientId = self.client)

        #p.setJointMotorControlArray(self.arm,self.joints,mode,forces = action,physicsClientId = self.client)
        


    def get_observation(self):
        ## CHANGE

        obs = []
        for i in self.joints:
            pos,vel,_,_ = p.getJointState(self.arm,i,physicsClientId = self.client)
            obs.append(pos)
            obs.append(vel)

        return obs