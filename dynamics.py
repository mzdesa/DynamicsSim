import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

class Dynamics:
    """
    Skeleton class for system dynamics
    Includes methods for returning state derivatives, plots, and animations
    """
    def __init__(self, x0, stateDimn, inputDimn, relDegree = 1):
        """
        Initialize a dynamics object
        Args:
            x0 (stateDimn x 1 numpy array): initial condition state vector
            stateDimn (int): dimension of state vector
            inputDimn (int): dimension of input vector
            relDegree (int, optional): relative degree of system. Defaults to 1.
        """
        self.stateDimn = stateDimn
        self.inputDimn = inputDimn
        self.relDegree = relDegree
        
        #store the state and input
        self._x = x0
        self._u = None
    
    def get_state(self):
        """
        Retrieve the state vector
        """
        return self._x
        
    def deriv(self, x, u, t):
        """
        Returns the derivative of the state vector
        Args:
            x (stateDimn x 1 numpy array): current state vector at time t
            u (inputDimn x 1 numpy array): current input vector at time t
            t (float): current time with respect to simulation start
        Returns:
            xDot: state_dimn x 1 derivative of the state vector
        """
        return np.zeros((self.state_dimn, 1))
    
    def integrate(self, u, t, dt):
        """
        Integrates system dynamics using Euler integration
        Args:
            u (inputDimn x 1 numpy array): current input vector at time t
            t (float): current time with respect to simulation start
            dt (float): time step for integration
        Returns:
            x (stateDimn x 1 numpy array): state vector after integrating
        """
        #integrate starting at x
        self._x = self.get_state() + self.deriv(self.get_state(), u, t)*dt
        return self._x
    
    def get_plots(self, x, u, t):
        """
        Function to show plots specific to this dynamic system.
        Args:
            x ((stateDimn x N) numpy array): history of N states to plot
            u ((inputDimn x N) numpy array): history of N inputs to plot
            t ((1 x N) numpy array): history of N times associated with x and u
        """
        pass
    
    def show_animation(self, x, u, t):
        """
        Function to play animations specific to this dynamic system.
        Args:
            x ((stateDimn x N) numpy array): history of N states to plot
            u ((inputDimn x N) numpy array): history of N inputs to plot
            t ((1 x N) numpy array): history of N times associated with x and u
        """
        pass
    
    
"""
**********************************
PLACE YOUR DYNAMICS FUNCTIONS HERE
**********************************
"""
class DoubleIntDyn(Dynamics):
    def __init__(self, x0 = np.zeros((6, 1)), stateDimn = 6, inputDimn = 3, relDegree = 2):
        """
        Init function for a Planar double integrator system.
        Inputs are Fx and Fy (no Fz)

        Args:
            x0 (_type_): _description_ (x, y, z, x_dot, y_dot, z_dot)
            stateDimn (_type_): _description_
            inputDimn (_type_): _description_
            relDegree (int, optional): _description_. Defaults to 2.
        """
        super().__init__(x0, stateDimn, inputDimn, relDegree)
        
        #store the linear system dynamics matrices
        self._A = np.hstack((np.zeros((6, 3)), np.vstack((np.eye(3), np.zeros((3, 1))))))
        self._B = np.vstack((np.zeros((3, 1)), np.eye(3)))
    
    def deriv(self, x, u, t):
        """
        Returns the derivative of the state vector
        Args:
            x (4 x 1 numpy array): current state vector at time t
            u (2 x 1 numpy array): current input vector at time t
            t (float): current time with respect to simulation start
        Returns:
            xDot: state_dimn x 1 derivative of the state vector
        """
        return self._A@x + self._B@u
        
    def get_state_spatial(self):
        """
        Returns the purely spatial (X, Y, Z) component of the state vector

        Returns:
            (2x1 numpy array): (x, y, z) position of the system
        """
        return self.get_state()[0:3].reshape((3, 1))
    
    def get_state_velocity(self):
        """
        Returns the velocity component (X_dot, Y_dot, Z_dot) of the state vector

        Returns:
            (2x1 numpy array): (x_dot, y_dot) velocity of the system
        """
        return self.get_state()[2:].reshape((2, 1))
    
    def get_rotation_matrix(self):
        """
        Returns the rotation matrix from the car frame to the spatial frame.
        """
        #use the velocity to get the first component.
        
    def show_animation(self, x, u, t):
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