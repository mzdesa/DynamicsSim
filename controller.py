import numpy as np

"""
File containing controllers 
"""
class Controller:
    def __init__(self, observer, lyapunov = None, trajectory = None, obstacleQueue = None, uBounds = None):
        """
        Skeleton class for feedback controllers
        Args:
            observer (Observer): state observer object
            lyapunov (LyapunovBarrier): lyapunov functions, LyapunovBarrier object
            trajectory (Trajectory): trajectory for the controller to track (could just be a constant point!)
            obstacleQueue (ObstacleQueue): ObstacleQueue object, stores all barriers for the system to avoid
            uBounds ((Dynamics.inputDimn x 2) numpy array): minimum and maximum input values to the system
        """
        #store input parameters
        self.observer = observer
        self.lyapunov = lyapunov
        self.trajectory = trajectory
        self.obstacleQueue = obstacleQueue
        self.uBounds = uBounds
        
        #store input
        self._u = None
    
    def eval_input(self, t):
        """
        Solve for and return control input
        Inputs:
            t (float): time in simulation
        Returns:
            u ((Dynamics.inputDimn x 1)): input vector, as determined by controller
        """
        self._u = np.zeros((self.observer.inputDimn, 1))
        return self._u
    
    def get_input(self):
        """
        Retrieves input stored in class parameter
        Returns:
            self._u: most recent input stored in class paramter
        """
        return self._u
    
class PlanarQrotorPD:
    def __init__(self, observer, lyapunov = None, trajectory = None, obstacleQueue = None, uBounds = None):
        """
        Init function for a planar quadrotor controller.

        Args:
            observer (Observer): state observer object
            lyapunov (LyapunovBarrier): lyapunov functions, LyapunovBarrier object
            trajectory (Trajectory): trajectory for the controller to track (could just be a constant point!)
            obstacleQueue (ObstacleQueue): ObstacleQueue object, stores all barriers for the system to avoid
            uBounds ((Dynamics.inputDimn x 2) numpy array): minimum and maximum input values to the system
        """
        super.__init__(observer, lyapunov, trajectory, obstacleQueue, uBounds)
        
        #Initialize variables for the gain parameters
        self.Kp = np.eye(3) #proportional position gain
        self.Kd = np.eye(3) #derivative position gain
        self.Ktheta = 1 #proportional orientation gain
        self.Komega = 1 #derivative orientation gain
        
        #Store quadrotor parameters from the observer
        self.m = self.observer.dynamics._m
        self.Ixx = self.observer.dynamics._Ixx
        self.g = 9.81 #store gravitational constant
        
        #store Euclidean basis vector
        self.e1 = np.array([[1, 0, 0]]).T
        self.e2 = np.array([[0, 1, 0]]).T
        self.e3 = np.array([[0, 0, 1]]).T
        
    def set_params(self, Kp, Kd, Ktheta, Komega):
        """
        Function to set controller gain parameters
        Args:
            Kp ((3x3) numpy array): proportional gain
            Kd ((3x3) numpy array): derivative gain
            Ktheta (float): proportional orientation gain
            Komega (flota): derivative orientation gain
        """
        self.Kp = Kp
        self.Kd = Kd
        self.Ktheta = Ktheta
        self.Komega = Komega
    
    def get_position_error(self, t):
        """
        Function to return the position error vector x_d - x_q
        Args:
            t (float): current time in simulation
        Returns:
            eX ((3 x 1) NumPy array): x_d - x_q based on current quadrotor state
        """
        #retrieve desired and current positions
        xD = self.trajectory.pos(t)
        xQ = self.observer.get_pos()
        
        #return difference
        return xD - xQ
    
    def get_velocity_error(self, t):
        """
        Function to return velocity error vector v_d - v_q
        Args:
            t (float): current time in simulation
        Returns:
            eX ((3 x 1) NumPY array): vD - vQ
        """
        #retrieve desired and current velocities
        vD = self.trajectory.vel(t)
        vQ = self.observer.get_vel()
        
        #return difference
        return vD - vQ
    
    def eval_force_vec(self, t):
        """
        Function to evaluate the force vector input to the system using point mass dynamics.
        Args:
            t (float): current time in simulation
        Returns:
            f ((3 x 1) NumPy Array): virtual force vector to be tracked by the orientation controller
        """
        #find position and velocity error
        eX = self.get_position_error(t)
        eV = self.get_velocity_error(t)
        
        #get desired acceleration from trajectory
        aD = self.trajectory.accel(t)
        
        #calculate control input - add feedforward acceleration term
        return self.Kp@eX + self.Kd@eV + self.m*self.g*self.e3 + self.m*aD
    
    def eval_desired_orient(self, t, f):
        """
        Function to evaluate the desired orientation of the system.
        Args:
            t (float): current time in simulation
            f ((3 x 1) NumPy array): force vector to track from point mass dynamics
        Returns:
            thetaD (float): desired angle of quadrotor WRT world frame
        """
        return np.atan2(f[2, 0], f[1, 0])
    
    def eval_orient_error(self, t):
        """
        Evalute the orientation error of the system thetaD - thetaQ
        Args:
            t (float): current time in simulation
        Returns:
            eOmega (float): error in orientation angle
        """
        f = self.eval_force(t)
        thetaD = self.eval_desired_orient(t, f)
        thetaQ = self.observer.get_orient()
        
        #return the difference
        return thetaD - thetaQ
    
    def eval_moment(self, t):
        """
        Function to evaluate the moment input to the system
        Args:
            t (float): current time in simulation
        Returns:
            M (float): moment input to quadrotor
        """
        eTheta = self.eval_orient_error(t)
        eOmega = 0 - self.observer.get_omega() #assume zero angular velocity desired
        
        #return the PD controller output - assume zero desired angular acceleration
        return self.Ktheta*eTheta + self.Komega*eOmega + self.Ixx*0
    
    def eval_force_scalar(self, t):
        """
        Evaluates the scalar force input to the system.
        Args:
            t (float): current time in simulation
        Returns:
            F (float): scalar force input from PD control
        """
        #first, construct R, a rotation matrix about the x axis
        thetaQ = self.observer.get_orient()
        R = np.array([[np.cos(thetaQ), 0, np.sin(thetaQ)], 
                      [0, 1, 0], 
                      [-np.sin(thetaQ), 0, np.cos(thetaQ)]])
        
        #find and return the scalar force
        return self.eval_force_vec(t).T@R@self.e3
        
    def get_input(self, t):
        """
        Get the control input F, M to the planar quadrotor system
        Args:
            t (float): current time in simulation
        Returns:
            F (float): scalar force input to system
            M (float): scalar moment input to system
        """
        return self.eval_force_scalar(t), self.eval_moment(t)