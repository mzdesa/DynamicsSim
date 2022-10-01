#Python Dependencies
import numpy as np
import gym
from gym import spaces


class Environment(gym.Env):
    def __init__(self, dynamics, controller, observer):
        """
        Initializes a simulation environment
        Args:
            dynamics (Dynamics): system dynamics object
            controller (Controller): system controller object
            observer (Observer): system state estimation object
        """
        #store system parameters
        self.dynamics = dynamics
        self.controller = controller
        self.observer = observer
        
        #Define gym environment parameters
        self.action_space = spaces.Box(low = -np.inf, high = np.inf, shape = (self.dynamics.inputDimn, ))
        self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape = (self.dynamics.stateDimn, ))
        self.iter = 0 #number of iterations
        self.t = 0 #time in seconds 
        self.reward = 0
        self.info = {}
        self.done = False
        
        #Store system state
        self.x = self.dynamics.get_state() #Actual state of the system
        self.x0 = self.x #store initial condition for use in reset
        self.xObsv = None #state as read by the observer
        
        #Define history arrays
        self.xHist = self.x #initialize with the initial parameter
        self.uHist = None
        self.tHist = None
        
        #Define simulation parameters
        self.SIM_FREQ = 1000 #integration frequency in Hz
        self.CONTROL_FREQ = 50 #control frequency in Hz
        self.SIMS_PER_STEP = self.SIM_FREQ/self.CONTROL_FREQ
        self.TOTAL_SIM_TIME = 10 #total simulation time in s
        
    def reset(self):
        """
        Reset the gym environment to its inital state.
        """
        #Reset gym environment parameters
        self.iter = 0 #number of iterations
        self.t = 0 #time in seconds
        self.reward = 0
        self.info = {}
        self.done = False
        
        #Reset system state
        self.x = self.x0 #retrieves initial condiiton
        self.xObsv = None #reset observer state
        
        #Define history arrays
        self.xHist = self.x
        self.uHist = None
        self.tHist = None
        
        #Define simulation constants
        self.SIM_FREQ = 1000 #integration frequency in Hz
        self.CONTROL_FREQ = 50 #control frequency in Hz
        self.SIMS_PER_STEP = self.SIM_FREQ//self.CONTROL_FREQ #integer divide
    
    def step(self, acs):
        """
        Step the sim environment by one integration
        Inputs:
            acs: sample from the action space (unused)
        """
        #retrieve current state information
        self._get_observation() #updates the observer
        
        #solve for the control input using the observed state
        self.controller.eval_input(self.xObsv, self.t)
        
        #integrate through that input over sim freq
        for i in range(self.SIMS_PER_STEP):
            self.dynamics.integrate(self.controler.get_input(), self.t, 1/self.SIM_FREQ) #integrate dynamics
            self.t += 1/self.SIM_FREQ #increment the time
            
        #update the deterministic system data, iterations, and history array
        self._update_data()
        
        #check if the simulation is complete
        self._is_done()
        
        #update reward (only implemented for reinforcement learning)
        self._get_reward()
        
        return self.x, self.reward, self.done, self.info
        
    
    def _update_data(self):
        """
        Update history arrays and deterministic state data
        """
        #update the actual state of the system
        self.x = self.dynamics.get_state()
        
        #update the number of iterations of the step function
        self.iter +=1
        
        #append the input, time, and state to their history queues
        self.xHist = np.hstack((self.xHist, self.x))
        self.uHist = np.hstack((self.uHist, self.controller.get_input()))
        self.tHist = np.hstack((self.tHist, self.t))
    
    def _get_observation(self):
        """
        Updates self.xObsv using the observer data
        """
        pass
    
    def _get_reward(self):
        """
        Calculate the total reward for ths system and update the reward parameter.
        Only implement for use for reinforcement learning.
        """
        return 0
    
    def _is_done(self):
        """
        Check if the simulation is complete
        Returns:
            boolean: whether or not the time has exceeded the total simulation time
        """
        #check current time with respect to simulation time
        if self.t >= self.TOTAL_SIM_TIME:
            return True
        return False
    
    def render(self):
        """
        Provide visualization of the environment
        """
        self.dynamics.show_animation(self.xHist, self.uHist, self.tHist)