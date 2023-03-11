import numpy as np
import casadi as ca
from state_estimation import *

"""
File containing controllers 
"""
class Controller:
    def __init__(self, observer, lyapunov = None, trajectory = None, obstacleQueue = None, uBounds = None):
        """
        Skeleton class for feedback controllers
        Args:
            dynamics (Dynamics): system Dynamics object
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
    
class TurtlebotFBLin:
    def __init__(self, observer, trajectory):
        """
        Class for a feedback linearizing controller for a single turtlebot within a larger system
        Args:
            observer (Observer): state observer object
            traj (Trajectory): trajectory object
        """
        #store input parameters
        self.observer = observer
        self.trajectory = trajectory

    def eval_input(self, t):
        """
        Solves for the control input to turtlebot i using a CBF-QP controller.
        Inputs:
            t (float): current time in simulation
            i (int): index of turtlebot in the system we wish to control (zero indexed)
        """
        #get the state vector of turtlebot i
        q = self.observer.get_state()

        #find the input
        self._u = ...
    
    def get_input(self):
        """
        Retrieves input stored in class parameter
        Returns:
            self._u: most recent input stored in class parameter
        """
        return self._u

class TurtlebotCBFQP:
    def __init__(self, observer, barriers, trajectory):
        """
        Class for a CBF-QP controller for a single turtlebot within a larger system.
        Args:
            observer (EgoTurtlebotObserver): state observer object for a single turtlebot within the system
            barriers (List of TurtlebotBarrier): List of TurtlebotBarrier objects corresponding to that turtlebot
            traj (Trajectory): trajectory object
        """
        #store input parameters
        self.observer = observer
        self.barriers = barriers
        self.trajectory = trajectory

        #create a nominal controller
        self.nominalController = TurtlebotFBLin(observer, trajectory)

        #store the input parameter (should not be called directly but with get_input)
        self._u = None

    def eval_input(self, t):
        """
        Solves for the control input to turtlebot i using a CBF-QP controller.
        Inputs:
            t (float): current time in simulation
            i (int): index of turtlebot in the system we wish to control (zero indexed)
        """
        #get the state vector of turtlebot i
        q = self.observer.get_state()

        #get the nominal control input to the system
        kX = self.nominalController.eval_input()

        #apply the N-1 barrier constraints

        self._u = ...
    
    def get_input(self):
        """
        Retrieves input stored in class parameter
        Returns:
            self._u: most recent input stored in class parameter
        """
        return self._u
    

class ControllerManager(Controller):
    def __init__(self, observerManager, barrierManager, trajectoryManager, controlType):
        """
        Class for a CBF-QP controller for a single turtlebot within a larger system.
        Managerial class that points to N controller instances for the system. Interfaces
        directly with the overall system dynamics object.
        Args:
            observer (Observer): state observer object
            controller (String): string of controller type 'TurtlebotCBFQP' or 'TurtlebotFBLin'
            sysTrajectory (SysTrajectory): SysTrajectory object containing the trajectories of all N turtlebots
            nominalController (Turtlebot_FB_Lin): nominal feedback linearizing controller for CBF-QP
        """
        #store input parameters
        self.observerManager = observerManager
        self.barrierManager = barrierManager
        self.trajectoryManager = trajectoryManager
        self.controlType = controlType

        #store the input parameter (should not be called directly but with get_input)
        self._u = None

        #get the number of turtlebots in the system
        self.N = self.observer.dynamics.N

        #create a controller dictionary
        self.controllerDict = {}

        #create N separate controllers - one for each turtlebot - use each trajectory in the trajectory dict
        for i in range(self.N):
            #extract the ith trajectory
            trajI = self.trajectoryManager.get_traj_i(i)

            #get the ith observer object
            egoObsvI = self.observerManager.get_observer_i(i)

            #get the ith barrier object
            barrierI = self.barrierManager.get_barrier_list_i(i)

            #create a controller using the three objects above - add the controller to the dict
            if self.controlType == 'TurtlebotCBFQP':
                #create a CBF QP controller
                self.controllerDict[i] = TurtlebotCBFQP(egoObsvI, barrierI, trajI)
            elif self.controlType == 'TurtlebotFBLin':
                #create a feedback linearizing controller
                self.controllerDict[i] = TurtlebotFBLin(egoObsvI, trajI)
            else:
                raise Exception("Invalid Controller Name Error")

    def eval_input(self, t):
        """
        Solve for and return control input for all N turtlebots. Solves for the input ui to 
        each turtlebot in the system and assembles all of the input vectors into a large 
        input vector for the entire system.
        Inputs:
            t (float): time in simulation
        Returns:
            u ((Dynamics.inputDimn x 1)): input vector, as determined by controller
        """
        #initilialize input vector as zero vector - only want to store once all have been updated
        u = np.zeros((self.observerManager.dynamics.inputDimn, 1))

        #loop over the system to find the input to each turtlebot
        for i in range(self.N):
            #solve for the latest input to turtlebot i, store the input in the u vector
            self.controllerDict[i].eval_input(t)
            u[2*i : 2*i + 2] = self.controllerDict[i].get_input()

        #store the u vector in self._u
        self._u = u

        #return the full force vector
        return self._u
    
    def get_input(self):
        """
        Retrieves input stored in class parameter
        Returns:
            self._u: most recent input stored in class paramter
        """
        return self._u