#Python Dependencies
import numpy as np

#our dependencies
from dynamics import *
from controller import *
from barrier_lyapunov import *
from state_estimation import *

#environment dependencies
import gym
from gym import spaces
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Environment(gym.Env):
    """
    Class to represent environment for reinforcement learning
    Inherits from gym.Env
    """
    def __init__(self, dynamics, observer, reward, policy, barrier, lyapunov, max_timesteps, action_bounds, q, queue_size = 10):
        """
        Init function for RL environment
        inputs:
        dynamics: Dynamics object
        observer: Observer object
        reward: Reward object
        policy: Policy object (cbf clf, etc)
        barrier: Barrier object
        lyapunov: Lyapunov object
        max_timesteps: max iterations for RL to run
        action_bounds: 2xinput_dimn array, min/max actions in each index
        q: initial condition for the state vector
        queue_size: size for queue history to be passed into neural network
        """
        #store init function inputs
        self.dynamics = dynamics
        self.observer = observer
        self.reward_obj = reward #reward object
        self.policy = policy
        self.barrier = barrier
        self.lyapunov = lyapunov
        self.max_timesteps = max_timesteps
        self.action_bounds = action_bounds
        
        #Define state information
        self.q = q #the state vector for DYNAMICS. NOT the same as s_t, the input to the neural network!!
        self.q0 = self.q #store the initial state for the reset
        self.u = np.zeros((self.dynamics.input_dimn, 1)) #define a current input vector
        
        #Define simulator properties
        self.sim_freq = 1000
        if type(self.dynamics) == DoubleIntegrator:
            self.exp_env_freq = 100 #use 100hz as control frequency for double integrator
        else:
            self.exp_end_freq = 30 #use 30hz as standard for other dynamical systems
        self.num_sims_per_env_step = self.sim_freq//self.exp_env_freq

        #define observation space
        obs_dimn = queue_size*(self.dynamics.input_dimn + 2*self.dynamics.state_dimn) #what's PASSED IN to the neural network! N*(q, q_dot, u) (u is control input)
        self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape = (obs_dimn, )) #obsv. space for state vector
        
        #define action space - bounds using action_bounds - shape from lie derivative shape
        action_dimn = 2*(1 + self.dynamics.input_dimn) #What's RETURNED by the neural network (lfv, lgv, lfh, lgh)
        self.action_space = spaces.Box(low = np.amin(action_bounds), high = np.amax(action_bounds), shape = (action_dimn, ))
        
        #define iterations so far to be 0
        self.iter = 0
        self.time_s = 0 #time in seconds
        
        #define the current reward
        self.reward = 0
        
        #define other Gym environment terms
        self.info = {}
        self.done = None
        
        #define FULL history arrays
        self.q_hist = q #state history
        self.a_hist = [] #action history
        self.r_hist = [self.reward] #reward history
        
        #define state + input queues to be passed into network
        self.queue_size = queue_size
        self.q_queue = deque([np.zeros((self.dynamics.state_dimn, 1))]*self.queue_size) #state history queue, size queue_size
        self.q_dot_queue = deque([np.zeros((self.dynamics.state_dimn, 1))]*self.queue_size) #state derivative history queue
        self.u_queue = deque([np.zeros((self.dynamics.input_dimn, 1))]*self.queue_size) #control input history queue
        
    def reset(self, z = False):
        """
        Function to reset the agent and environment to its initial condition
        Think of this like the init method - should be called before training.
        Inputs:
        z: whether we're using Zhongyu's code (True) or stable baselines (false)
        """
        pass
    
    
    def step(self, lie_derivs, z = False):
        """
        Step function for RL simulator
        Simulates a full step of the dynamics in time
        Inputs:
        lie_derivs: lie derivatives for control policy. [lfv, lgv, lfh, lgh] -> this is our "action" a_t!
        z: whether it's Zhongyu's PPO or SB3
        Returns:
        state observation (within observation space), reward, done (True or False, reached endpoint), info (dict for bug fixing)
        """
        pass
    
    def _update_data(self, q, u, r = None, step = True, hist_update = False):
        """
        Update function to update the simulation parameters
        Inputs:
        q: current state
        a: current action
        r: reward
        step: whether a step has occurred or not
        """
        pass
    
    def _apply_perturbation(self):
        """
        Function to apply a perturbation directly to the obstacle state
        Applies a small random offset to the state vector
        """
        pass
    
    def _get_observation(self, u, queue_step = True):
        """
        Function to perform measurement of the current state
        Inputs:
        u: the input vector we pass into the system as a control input
        Returns:
        potentially stochastic estimate of current state, based on deterministic actual state
        """
        pass
    
    def _get_reward(self, s_t_obs, lie_deriv, delta = 0):
        """
        Function to calculate the reward given a sequence of actions (could also be a single action)
        Inputs:
        s_t_obs: state (sequence or single vector), input_dimn x N numpy array
        lie_deriv: estimate of lie derivatives from neural network (a_t)
        delta: value of slack variable in optimization
        """
        return 0
    
    def _is_done(self, h, failed = False):
        """
        Function to check if the simulation has finished
        h: barrier function value, scalar
        failed: whether the optimization failed or not
        """
        return False
    
    def render(self, type = "acc"):
        """_
        Provide visualization of the environment
        Inputs:
        type: system to animate, default is adaptive cruise
        """
        if type == "integrator":
            fig, ax = plt.subplots()
            # set the axes limits
            ax.axis([0, 10, -1,1])
            # create a point in the axes
            point, = ax.plot(0,1, marker="o")
            num_frames = self.q_hist.shape[1]-1
            #define animation callback
            def animate(i):
                #plot the distance to the car ahead on the x axis
                x = self.q_hist[0, i] #extract the x position
                #keep the y position of the car as zero
                y = 0
                point.set_data(x, y)
                return point,
            #Plot the obstacle car at (10, 0)
            ax.scatter([10], [0], color = 'r')
            #crete animation
            anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=self.dynamics.dt*1000, blit=True)
            plt.xlabel("Car Position (m)")
            plt.title("Distance to Car Ahead")
            plt.show()
            
            #Now, plot the actual position versus the desired position to show effect of lyapunov function
            y_data = self.q_hist[0, :].tolist() #extract all of the velocity data to plot on the y axis
            x_data = np.linspace(0, 10, len(y_data)) #time
            goal_pos = self.lyapunov.x_g[0, 0] #extract the goal velocity
            goal_data = [goal_pos]*len(y_data)
            
            #now, plot the velocities against time
            plt.plot(x_data, y_data)
            plt.plot(x_data, goal_data)
            plt.xlabel("Time (s)")
            plt.ylabel("Position (m)")
            plt.title("Car Position")
            plt.show()