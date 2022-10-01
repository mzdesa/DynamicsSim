import numpy as np

class LyapunovBarrier:
    """
    Skeleton class for Lyapunov/Barrier functions.
    Includes utilities to get Lyapunov/Barrier values and Lyapunov/Barrier derivatives
    """
    def __init__(self, stateDimn, inputDimn, dynamics):
        """
        Init function for a Lyapunov/Barrier object
        Args:
            stateDimn (int): length of state vector
            inputDimn (int): length of input vector
            dynamics (Dynamics): dynamics object
        """
        self.stateDimn = stateDimn
        self.inputDimn = inputDimn
        self.dynamics = dynamics
        
        #parameters to store values
        self._vals = None #stored derivatives + value of function (initialize as None)
    
    def eval(self, x, u, t):
        """
        Returns a list of derivatives (going until zeroth derivative) of the lyapunov/Barrier function.
        Args:
            x (numpy array, (input_dimn x 1)): current state vector of system
            u (numpy array, (input_dimn x 1)): current input vector to system
            t (float): current time with respect to simulation state
        Returns:
        [..., vDDot, vDot, v] ((self.dynamics.relDegree + 1, 1) numpy Array): dynamics.relDegree lyapunov/Barrier time derivs, Descending order
        """
        self._vals = np.zeros((self.dynamics.relDegree + 1, 1))
        return self._vals
    
    def get(self):
        """
        Retreives stored function and derivative values
        Returns:
        [..., vDDot, vDot, v] ((self.dynamics.relDegree + 1, 1) numpy Array): dynamics.relDegree lyapunov/Barrier time derivs, Descending order
        """
        return self._vals

"""
********************************
ADD YOUR LYAPUNOV FUNCTIONS HERE
********************************
"""

class DoubleIntLyapunov(LyapunovBarrier):
    def __init__(self, stateDimn, inputDimn, dynamics):
        """
        Double integrator system Lyapunov function.
        Args:
            stateDimn (int): length of state vector
            inputDimn (int): length of input vector
            dynamics (Dynamics): 
        """
        super().__init__(stateDimn, inputDimn, dynamics)
    
    def eval(self, x, u, t):
        return super().eval(x, u, t)
    
"""
********************************
ADD YOUR BARRIER FUNCTIONS HERE
********************************
"""

class DoubleIntBarrier(LyapunovBarrier):
    def __init__(self, stateDimn, inputDimn, dynamics):
        """
        Double integrator system Lyapunov function.
        Args:
            stateDimn (int): length of state vector
            inputDimn (int): length of input vector
            dynamics (Dynamics): 
        """
        super().__init__(stateDimn, inputDimn, dynamics)
    
    def eval(self, x, u, t):
        return super().eval(x, u, t)