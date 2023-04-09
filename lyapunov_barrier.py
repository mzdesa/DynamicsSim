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
    def __init__(self, stateDimn, inputDimn, dynamics, observerEgo, observerObstacle, buffer):
        """
        Double integrator system Lyapunov function.
        Args:
            stateDimn (int): length of (entire) state vector
            inputDimn (int): length of input vector
            dynamics (Dynamics): dynamics object for the entire turtlebot system
            observerEgo (EgoObserver): observer object for the turtlebot we're deciding the input to
            observerObstacle (EgoObserver): observer object for an obstacle
        """
        super().__init__(stateDimn, inputDimn, dynamics)
        self._barrierPt = None
        self._buffer = buffer #barrier buffer
        self.observerEgo = observerEgo #store the system observer
        self.observerObstacle = observerObstacle #store an observer for the obstacle

        #define the radius of the turtlebot
        self.rt = 0.3
    
    def eval(self, u, t):
        """
        Evaluate the Euclidean distance to the barrier point.
        Args:
            u (input_dimn x 1 numpy array): input vector
        Returns:
            (List): cbf time derivatives
        """
        #get the position and velocity of the ego and the obstacle objects
        qe = self.observerEgo.get_state()

        #calculate qeDot from system dynamics (Not from observer ego)
        phi = qe[2, 0]
        qeDot = np.array([[np.cos(phi), 0], [np.sin(phi), 0], [0, 1]])@u

        #get the obstacle states from the observer
        qo = self.observerObstacle.get_state()
        qoDot = self.observerObstacle.get_vel()

        #evaluate the CBF
        h = (qe[0, 0] - qo[0, 0])**2 + (qe[1, 0] - qo[1, 0])**2 - (2*self.rt)**2

        #evaluate the derivative of the CBF
        hDot = 2*(qe[0, 0] - qo[0, 0])*((qeDot[0, 0] - qoDot[0, 0])) + 2*(qe[1, 0] - qo[1, 0])*((qeDot[1, 0] - qoDot[1, 0]))
        
        #return the two derivatives and the barrier function
        self._vals = [h, hDot]
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
            observer (ObserverManager): observer object for the whole turtlebot system
        """
        #store the number of turtlebots in the system
        self.N = N

        #create a barrier function dictionary that stores the N - 1 barriers for each turtlebot
        self.barrierDict = {}

        #store the observer manager
        self.observerManager = observer

        #create a set of N - 1 Barrier functions for that turtlebot
        for i in range(self.N):
            #create the N - 1 turtlebots - one for all indices other than i
            indexList = list(range(N))

            #remove the ego turtlebot index i from this list
            indexList.remove(i)

            #initialize an empty list to contain the barrier functions for the ego turtlebot
            egoBarrierList = []

            #get the observer corresponding to the ego turtlebot
            observerEgo = self.observerManager.get_observer_i(i)

            #loop over the remaining indices and create a barrier function for each obstacle turtlebot
            for j in indexList:
                #ego index is i, obstacle index is j
                observerObstacle = self.observerManager.get_observer_i(j) #get the obstacle observer

                #append a barrier function with the obstacle and ego observers
                egoBarrierList.append(TurtlebotBarrier(stateDimn, inputDimn, dynamics, observerEgo, observerObstacle, buffer))

            #store the ego barrier list in the barrier function dictionary for the robots
            self.barrierDict[i] = egoBarrierList

    def get_barrier_list_i(self, i):
        """
        Function to retrieve the list of barrier function objects corresponding to turtlebot i
        Inputs:
            i (int): index of turtlebot (zero-indexed)
        Returns:
            barrierList (TurtlebotBarrier List): List of TurtlebotBarrier objects corresponding to turtlebot i
        """
        return self.barrierDict[i]
