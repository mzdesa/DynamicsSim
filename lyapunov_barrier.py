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
    
    def eval(self, u, t):
        """
        Returns a list of derivatives (going until zeroth derivative) of the lyapunov/Barrier function.
        Args:
            u (numpy array, (input_dimn x 1)): current input vector to system
            t (float): current time in simulation
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
ADD YOUR BARRIER FUNCTIONS HERE
********************************
"""

class TurtlebotBarrier(LyapunovBarrier):
    def __init__(self, iEgo, iObstacle, stateDimn, inputDimn, dynamics, observer, buffer):
        """
        Double integrator system Lyapunov function.
        Args:
            iEgo (int): index of the ego turtlebot (zero-indexed)
            iObstacle (int): index of the obstacle turtlebot (zero-indexed)
            stateDimn (int): length of (entire) state vector
            inputDimn (int): length of input vector
            dynamics (Dynamics): dynamics object for the entire turtlebot system
            observer (EgoObserver): observer object for a single turtlebot in the system
        """
        super().__init__(stateDimn, inputDimn, dynamics)
        self.iEgo = iEgo
        self.iObstacle = iObstacle
        self._barrierPt = None
        self._buffer = buffer #barrier buffer
        self.observer = observer #store the system observer
        
    def set_barrier_pt(self, pt):
        """
        Function to update the point used in the barrier function (in the world frame)
        Args:
            pt (3 x 1 numpy array): new point to be used for a barrier function, (x, y, z) position
        """
        self._barrierPt = pt
    
    def get_barrier_pt(self):
        """
        Retreive the barrier point from the class attribute
        """
        return self._barrierPt
    
    def eval(self, u, t):
        """
        Evaluate the Euclidean distance to the barrier point.
        Args:
            u (input_dimn x 1 numpy array): current input vector
        Returns:
            (List): cbf time derivatives
        """
        #first, get the spatial and velocity vectors from the observer
        x = self.observer.get_pos()
        v = self.observer.get_vel()
        
        #evaluate the barrier function value
        h = ((x - self._barrierPt).T@(x - self._barrierPt))[0, 0] - self._buffer**2
        
        #evaluate its first derivative - assume a still obstacle
        hDot = (2*v.T@(x - self._barrierPt))[0, 0]
        
        #evaluate its second derivative - assume double integrator point mass dynamics
        xddot = u #pull directly from the force vector, double integrator system
        hDDot = 2*(xddot.T@(x - self._barrierPt) + v.T@v)
        
        #return the two derivatives and the barrier function
        self._vals = [hDDot, hDot, h]
        return self._vals

class BarrierManager:
    def __init__(self, N, stateDimn, inputDimn, dynamics, observer, buffer):
        """
        Class to organize barrier functionf or a system of N turtlebots.
        Initializes N-1 TurtlebotBarrier objects for each of the N turtlebots in the system.
        Provides utilities for returning the N-1 barrier functions and their derivatives for each turtlebot.
        Inputs:
            stateDimn (int): length of state vector
            inputDimn (int): length of input vector
            dynamics (Dynamics): dynamics object for the whole turtlebot system
            observer (DoubleIntObserver): observer object for the whole turtlebot system
        """
        #store the number of turtlebots in the system
        self.N = N

        #create a barrier function dictionary that stores the N - 1 barriers for each turtlebot
        self.barrierList = {}

        #create a set of N - 1 Barrier functions for that turtlebot
        for i in range(self.N):
            #create the N - 1 turtlebots - one for all indices other than i
            indexList = list(range(N))

            #remove the ego turtlebot index i from this list
            indexList.remove(i)

            #initialize an empty list to contain the barrier functions for the ego turtlebot
            egoBarrierList = []

            #get the observer corresponding to that turtlebot
            egoObsvI = self.observerManager.get_observer_i(i)

            #loop over the remaining indices and create a barrier function for each obstacle turtlebot
            for j in indexList:
                #ego index is i, obstacle index is j
                egoBarrierList.append(TurtlebotBarrier(i, j, stateDimn, inputDimn, dynamics, egoObsvI, buffer))

            #store the ego barrier list in the barrier function dictionary for the robots
            self.barrierList[i] = egoBarrierList

    def get_barrier_list_i(self, i):
        """
        Function to retrieve the list of barrier function objects corresponding to turtlebot i
        Inputs:
            i (int): index of turtlebot (zero-indexed)
        Returns:
            barrierList (TurtlebotBarrier List): List of TurtlebotBarrier objects corresponding to turtlebot i
        """
        return self.barrierList[i]
