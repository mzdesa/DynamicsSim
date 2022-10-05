import numpy as np
import time
import casadi as ca

class Trajectory:
    """
    Class to define basic tracking trajectories in R3 for quadrotors
    Generates a smooth straight line trajectory with zero start and end velocity.
    """
    def __init__(self, start, end, T):
        """
        Init function for Trajectory class
        Inputs:
        start: starting position (3x1 numpy array)
        end: ending position (3x1 numpy array)
        T: total desired time for trajectory, in seconds 
        """
        self.x_o = start
        self.x_f = end
        self.T = T
        self.t_0 = None #keep track of the time of initialization of the trajectory
        self.first_step = True #Initialize a variable to check if this is the first step in the trajectory
        self.published_counter = 0
        self.prev_t = 0

    def pos(self, t):
        """
        Function to get desired position at time t
        Inputs:
        t: current time
        Returns:
        (x, y, z) coordinates for the quadrotor to track at time t
        """
        #use sinusoidal interpolation to get a smooth trajectory with zero velocity at endpoints
        if t>self.T:
            #if beyond the time of the trajectory end, return the desired position as a setpoint
            return self.x_f
        des_pos = (self.x_f-self.x_o)/2*np.sin(t*np.pi/self.T - np.pi/2)+(self.x_o+self.x_f)/2
        return des_pos #calculates all three at once
    
    def vel(self, t):
        """
        Function to get the desired velocity at time t
        Inputs:
        t: current time
        Returns:
        (v_x, v_y, v_z) velocity for the quadrotor to track at time t
        """
        #differentiate position
        if t>self.T:
            #If beyond the time of the trajectory end, return 0 as desired velocity
            return np.zeros((3, 1))
        des_vel = (self.x_f-self.x_o)/2*np.cos(t*np.pi/self.T - np.pi/2)*np.pi/self.T
        return des_vel

    def accel(self, t):
        """
        Function to get the desired acceleration at time t
        Inputs:
        t: current time
        Returns:
        (a_x, a_y, a_z) acceleration for the quadrotor to track at time t
        """
        #differentiate acceleration
        if t>self.T:
            #If beyond the time of the trajectory end, return 0 as desired acceleration
            return np.zeros((3, 1))
        des_accel = -(self.x_f-self.x_o)/2*np.sin(t*np.pi/self.T - np.pi/2)*(np.pi/self.T)**2
        return des_accel

    def get_state(self):
        """
        Function to get the desired state at a time t
        Inputs:
        t: current time
        Returns:
        x_d, v_d, a_d: desired position, velocity, and acceleration at time t
        """
        if self.first_step:
            #Set the start time with respect to time library if never run before
            self.t_0 = time.time() #get the time from utc
            #change the first step variable to false
            self.first_step = False
            
        #take the current time with respect to the start time
        t = time.time() - self.t_0
        # print("time: ", t)
        freq = 1/(t - self.prev_t)
        self.prev_t = t

        # print("Freq: ", freq)

        #generate the desired states using the time relative to start
        x_d, v_d, a_d = self.pos(t), self.vel(t), self.accel(t)
        return x_d, v_d, a_d