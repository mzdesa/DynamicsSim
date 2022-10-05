import numpy as np
import casadi as ca
from collections import deque

"""
File containing controllers 
"""
class Controller:
    def __init__(self, dynamics = None, observer = None, lyapunov = None, barrierQueue = None, uBounds = None):
        """
        Skeleton class for feedback controllers
        Args:
            dynamics (Dynamics): system Dynamics object
            observer (Observer): state observer object
            lyapunov (LyapunovBarrier): lyapunov functions, LyapunovBarrier object
            barrierQueue (BarrierQueue): BarrierQueue object, stores all barriers for the system to avoid
            uBounds ((Dynamics.inputDimn x 2) numpy array): minimum and maximum input values to the system
        """
        #store input parameters
        self.dynamics = dynamics
        self.observer = observer
        self.lyapunov = lyapunov
        self.barrierQueue = barrierQueue
        
        #store input
        self._u = None
    
    def eval_input(self):
        """
        Solve for and return control input
        Args:
            x ((Dynamics.stateDimn x 1) NumPy array): current state vector
            t (float): current time in simulation
        Returns:
            u ((Dynamics.inputDimn x 1)): input vector, as determined by controller
        """
        return np.zeros((self.inputDimn, 1))
    
    def get_input(self):
        """
        Retrieves input stored in class parameter
        Returns:
            self._u: most recent input stored in class paramter
        """
        return self._u

class CLF_CBF_QP(Controller):
    def __init__(self, dynamics = None, observer = None, lyapunov = None, barrierQueue = None, uBounds = None):
        """
        Skeleton class for feedback controllers
        Args:
            dynamics (Dynamics): system Dynamics object
            observer (Observer): state observer object
            lyapunov (LyapunovBarrier): lyapunov functions, LyapunovBarrier object
            barrierQueue (BarrierQueue): BarrierQueue object, stores all barriers for the system to avoid
            uBounds ((Dynamics.inputDimn x 2) numpy array): minimum and maximum input values to the system
        """
        #store input parameters
        self.dynamics = dynamics
        self.observer = observer
        self.lyapunov = lyapunov
        self.barrierQueue = barrierQueue
        
        #store input
        self._u = None
        
        #store controller parameters
        self._useCBF = True
        self._useCLF = True
        self._useOptiSmoothing = False
        
        #store optimization tuning parameters - number of CBF tuning constants equal the relative degree + 1
        self._cbfAlphas = np.ones((1, self.dynamics.relDegree+1))
        self._clfGammas = np.ones((1, self.dynamics.relDegree+1))
        
    def set_params(self, useCBF, useCLF, useOptiSmoothing, cbfAlphas, clfGammas):
        """
        Set the parameters for the controller
        Args:
            useCBF (_type_): use the CBF constraint in the optimization
            useCLF (_type_): use the CLF constraint in the optimization
            useOptiSmoothing (_type_): use previous input smoothing in the optimization
            cbfAlphas ((1, self.Dynamics.relDegre + 1) numpy Array): Matrix of CBF tuning constaints in descending order of derivative
            clfGammas ((1, self.Dynamics.relDegre + 1) numpy Array): Array of CLF tuning constaints in descending order of derivative
        """
        #store optimization "On/Off switches"
        self._useCBF = useCBF
        self._useCLF = useCLF
        self._useOptiSmoothing = useOptiSmoothing
        
        #store optimization tuning parameters - number of CBF tuning constants equal the relative degree + 1
        self._cbfAlphas = cbfAlphas
        self._clfGammas = clfGammas
        
    def get_input(self):
        """
        Retrieve the input from the class parameter
        """
        return self._u
        
    def eval_input(self, t):
        """
        Solve for and return control input using CBF CLF QP
        Args:
            t (float): current time in simulation
        Returns:
            u ((Dynamics.inputDimn x 1)): input vector, as determined by controller
        """
        #extract state vector from observer
        x = self.observer.get_state()
        
        #set up Casadi optimization
        opti = ca.Opti()
        
        #set up optimization variables
        u = opti.variable(self.input_dimn, 1)
        delta = opti.variable()
            
        #Enforce CLF constraint
        if self._useCLF:
            #Apply matrix multiplication to evaluate the constraint
            opti.subject_to((self._clfGammas @ self.lyapunov.eval(x, u, t))[0, 0] <= delta)

        #Enforce CBF constraint
        if self._useCBF:
            #First update the barrier queue
            self.barrierQueue.update_queue()
            
            #iterate through the list of CBFs
            for cbf in self.barrierQueue.barriers:
                #Apply matrix multiplication to evaluate the constraint - extract each row of tuning constants
                opti.subject_to((self._cbfAlphas @ cbf.eval(x, u, t))[0, 0] >= 0)

        #Define Cost Function
        H = np.eye(self.input_dimn)
        cost = ca.mtimes(u.T, ca.mtimes(H, u)) + self.p*delta**2

        #set up optimization problem
        opti.minimize(cost)
        option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
        opti.solver("ipopt", option)

        #solve optimization
        try:
            sol = opti.solve()
            uOpt = sol.value(u) #extract optimal input
            solverFailed = False
        except:
            print("Solver failed!")
            solverFailed = True
            uOpt = np.zeros((self.input_dimn, 1))
        
        #store output in class param
        self._u = uOpt.reshape((self.input_dimn, 1))
        
        #return result
        return self._u, solverFailed
    
class CBF_QP(Controller):
    """
    Class for a CBF-QP Controller. Requires a nominal controller to be passed in.
    """
    def __init__(self, dynamics = None, observer = None, barrierQueue = None, uBounds = None, refController = None):
        """
        Initialize a CBF QP controller
        Args:
            dynamics (Dynamics): system Dynamics object
            observer (Observer): state Observer object
            barrierQueue (BarrierQueue): BarrierQueue object, stores all barriers for the system to avoid
            uBounds ((Dynamics.inputDimn x 2) numpy array): minimum and maximum input values to the system
            refController (Controller object): reference controller for use in CBF QP control
        """
        #store input parameters
        self.dynamics = dynamics
        self.observer = observer
        self.barrierQueue = barrierQueue
        self.refController = refController
        
        #store input
        self._u = None
        
        #store controller parameters
        self._useCBF = True
        self._useOptiSmoothing = False
        
        #store optimization tuning parameters - number of CBF tuning constants equal the relative degree + 1
        self._cbfAlphas = np.ones((1, self.dynamics.relDegree+1))
        
    def set_params(self, useCBF, useOptiSmoothing, cbfAlphas):
        """
        Set the parameters for the controller
        Args:
            useCBF (_type_): use the CBF constraint in the optimization
            useOptiSmoothing (_type_): use previous input smoothing in the optimization
            cbfAlphas ((1, self.Dynamics.relDegre + 1) numpy Array): Matrix of CBF tuning constaints in descending order of derivative
        """
        #store optimization "On/Off switches"
        self._useCBF = useCBF
        self._useOptiSmoothing = useOptiSmoothing
        
        #store optimization tuning parameters - number of CBF tuning constants equal the relative degree + 1
        self._cbfAlphas = cbfAlphas
    
    def eval_input(self, t):
        """
        Solve for and return control input using CBF QP
        Args:
            t (float): current time in simulation
        Returns:
            u ((Dynamics.inputDimn x 1)): input vector, as determined by controller
        """
        #extract state vector from observer
        x = self.observer.get_state()
        
        #set up Casadi optimization
        opti = ca.Opti()
        
        #set up optimization variables
        u = opti.variable(self.input_dimn, 1)
        
        #Solve for reference control input
        uRef = self.refController.eval_input(x, t)

        #Enforce CBF constraint
        if self._useCBF:
            #First update the barrier queue
            self.barrierQueue.update_queue()
            
            #iterate through the list of CBFs
            for cbf in self.barrierQueue.barriers:
                #Apply matrix multiplication to evaluate the constraint - extract each row of tuning constants
                opti.subject_to((self._cbfAlphas @ cbf.eval(x, u, t))[0, 0] >= 0)

        #Define Cost Function
        H = np.eye(self.input_dimn)
        cost = ca.mtimes((u-uRef).T, ca.mtimes(H, (u-uRef)))

        #set up optimization problem
        opti.minimize(cost)
        option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
        opti.solver("ipopt", option)

        #solve optimization
        try:
            sol = opti.solve()
            uOpt = sol.value(u) #extract optimal input
            solverFailed = False
        except:
            print("Solver failed!")
            solverFailed = True
            uOpt = np.zeros((self.input_dimn, 1))

        #store input in class param
        self._u = uOpt.reshape((self.input_dimn, 1))
        
        #return result
        return self._u, solverFailed
    
class StateFB(Controller):
    def __init__(self):
        """
        Initialize a state feedback controller
        """
        #store input
        self._u = None
        
        #store controller parameters
        self._xG = None #goal state
        self.K = None #controller gain matrix
        
    def set_params(self, xG, K):
        """
        Set the parameters for the controller
        Args:
            xG ((Dynamics.StateDimn x 1) numpy array): goal state for FB controller
            K ((Dynamics.InputDimn x Dynamics.StateDimn) numpy array): gain matrix for state feedback
        """
        self._xG = xG
        self.K = K
    
    def eval_input(self, x, t):
        """
        Solve for and return control input using state feedback
        Args:
            x ((Dynamics.stateDimn x 1) NumPy array): current state vector
            t (float): current time in simulation
        Returns:
            u ((Dynamics.inputDimn x 1)): input vector, as determined by controller
        """
        self._u = self.K@(self._xG - x)
        return self._u