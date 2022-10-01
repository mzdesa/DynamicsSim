import numpy as np

class Dynamics:
    """
    Skeleton class for system dynamics
    Includes methods for returning state derivatives, plots, and animations
    """
    def __init__(self, stateDimn, inputDimn, relDegree = 1):
        """
        Init function for a dynamics object
        """
        self.stateDimn = stateDimn
        self.inputDimn = inputDimn
        self.relDegree = relDegree
        
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