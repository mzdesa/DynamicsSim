import numpy as np
import casadi as ca
from collections import deque

"""
File containing controllers 
"""
class Controller:
    def __init__(self, dynamics = None, lyapunov = None, barrier = [], uBounds = None):
        """
        Skeleton class for feedback controllers
        Args:
            dynamics (Dynamics): system Dynamics object
            lyapunov (LyapunovBarrier): lyapunov functions, LyapunovBarrier object
            barrier (List of LyapunovBarrier): Python List of barrier functions, LyapunovBarrier objects
            uBounds ((Dynamics.inputDimn x 2) numpy array): minimum and maximum input values to the system
        """
        #store input parameters
        self.dynamics = dynamics
        self.lyapunov = lyapunov
        self.barrier = barrier
        
        #store input
        self._u = None
    
    def eval_input(self, x, t):
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
    def __init__(self, dynamics = None, lyapunov = None, barrier = [], uBounds = None):
        """
        Skeleton class for feedback controllers
        Args:
            dynamics (Dynamics): system Dynamics object
            lyapunov (LyapunovBarrier): lyapunov functions, LyapunovBarrier object
            barrier (List of LyapunovBarrier): Python List of barrier functions, LyapunovBarrier objects
            uBounds ((Dynamics.inputDimn x 2) numpy array): minimum and maximum input values to the system
        """
        #store input parameters
        self.dynamics = dynamics
        self.lyapunov = lyapunov
        self.barrier = barrier
        
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
        
        
    def eval_input(self, x, t):
        """
        Function to get control input given current state according to CLF CBF QP
        Uses Casadi for optimization
        Inputs:
        x_t: state vector at current time step, (state_dimn x 1) numpy vector
        lie_derivs: directly input the lie derivatives [lfv, lgv, lfh, lgh] as a python list
        feasible: use pointwise feasability approach
        u_queue: queue of previous N inputs
        """
        #update the lie derivtive parameters based on the current state

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
            #iterate through the list of CBFs
            for cbf in self.barrier:
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

        #return the optimal input!
        if isinstance(uOpt, float):
            #if input is a float, turn into a 1x1 numpy array
            return np.array([[uOpt]]), solverFailed
        return uOpt.reshape((self.input_dimn, 1)), solverFailed
    
class CBF_QP(Controller):
    """
    Class for a CBF-QP Controller. Requires a nominal controller to be passed in.
    """
    def __init__(self, cbf, state_dimn, input_dimn, alpha, nominal_controller, u_bounds = None):
        """
        Init function for a CBF QP Controller
        Inputs:
        cbf: CBF object
        state_dimn: dimension of state vector
        input_dimn: dimension of input vector
        alpha: CBF-QP constraint parameter
        nominal_controller: nominal controller for CBF-QP to compute min norm, controller object
        u_bounds: input bounds
        """
        #assign parameters from inputs
        if u_bounds is not None:
            self.u_min = u_bounds[0]
            self.u_max = u_bounds[1]
        else:
            #If u_bounds = None, set input bounds using inf
            self.u_min = np.ones((2, 1))*np.inf*-1
            self.u_max = np.ones((2, 1))*np.inf
            
        self.cbf = cbf
        self.state_dimn = state_dimn
        self.input_dimn = input_dimn
        self.alpha = alpha
        self.p_alpha = 10**8

        #store nominal controller
        self.nominal_controller = nominal_controller
        
        #Lie derivative parameters - store the current lie derivatives in here
        self.lfh = 0
        self.lgh = 0
    
    def eval_lie_derivs(self, x_t):
        """
        Function to evaluate lie derivatives and populate class attribute
        """
        self.lfh = self.cbf.eval_lfh(x_t)
        self.lgh = self.cbf.eval_lgh(x_t)
    
    def eval_input(self, x_t, lie_derivs = None, feasible = True, u_queue = None):
        """
        Function to get control input given current state according to CLF CBF QP
        Uses Casadi for optimization
        Inputs:
        x_t: state vector at current time step, (state_dimn x 1) numpy vector
        lie_derivs: directly input the lie derivatives [lfv, lgv, lfh, lgh] as a python list
        feasible: use pointwise feasability approach
        u_queue: queue of previous N inputs
        """
        #set up variables for easier access to class parameters
        x_t = x_t.reshape((x_t.shape[0], 1)) #ensure x_t has correct dimensions
        h = self.cbf.eval_h(x_t) #cbf function value
        buffer = 0.45
        h -= buffer

        #update the lie derivtive parameters based on the current state
        if lie_derivs:
            if len(lie_derivs) == 4:
                self.lfv, self.lgv, self.lfh, self.lgh = lie_derivs #unpack the lie derivative input array if provided
            else:
                #unpack additional terms for higher order cbf
                self.lfv, self.lgv, self.lfh, self.lgh, lf2v, lglfv, lf2h, lglfh = lie_derivs
        else:
            self.eval_lie_derivs(x_t) #else, use the eval functions from the clf and cbf

        #set up Casadi optimization
        opti = ca.Opti()
        #set up optimization variables
        u = opti.variable(self.input_dimn, 1) #input variable FOR NOW only 1x1
        alpha = self.alpha #tunable constant
        
        if feasible:
            delta_alpha = opti.variable()
            soft_alpha = self.alpha + delta_alpha
            
        #Enforce CBF constraint
        use_cbf = True #turn the cbf on/off for debugging
        if use_cbf:
            if (lie_derivs is not None and len(lie_derivs) == 4) or lie_derivs is None:
                #Apply relative degree 1 formulation
                if feasible:
                    opti.subject_to(self.lfh + ca.mtimes(self.lgh, u) >= -soft_alpha * h)
                else:
                    opti.subject_to(self.lfh + ca.mtimes(self.lgh, u) >= -alpha*h)
            else:
                #Apply relative degree 2 formulation
                opti.subject_to(lf2h + self.lfh + ca.mtimes(lglfh + self.lgh, u) >= -alpha*h)

        #Set up cost function
        input_filtering = True #add time-based filtering for the cost function
        if input_filtering and u_queue is not None:
            #form the input weights
            a = 0.75 #exponential weighting constant
            input_weights = np.array([[np.exp(-a*i) for i in range(len(u_queue))]])
            input_weights = (input_weights/np.sum(input_weights)).T #normalize to sum to 1
            
            #unpack the input queue:
            u_matrix = u_queue[0]
            for i in range(1,len(u_queue)):
                u_matrix = np.hstack((u_matrix, u_queue[i]))
            #multiply by weight matrix
            u_prev_weighted = u_matrix@input_weights
            filter_const = 1 #constant in cost function
        else:
            u_prev_weighted = np.zeros((self.input_dimn, 1))
            filter_const = 0 #set the weight of the filter to be 0        

        #get input from nominal controller
        k_x = self.nominal_controller.get_action(x_t)
        
        #Set up cost function
        p_alpha = self.p_alpha
        input_cost = ca.mtimes((u - k_x).T, (u - k_x)) + filter_const*ca.mtimes((u - u_prev_weighted).T, (u-u_prev_weighted)) #add filtering cost
        if feasible:
            #apply the feasibility cost function
            cost = input_cost + p_alpha*delta_alpha**2
        else:
            cost = input_cost

        #set up optimization problem
        opti.minimize(cost)
        option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
        opti.solver("ipopt", option)

        #solve optimization
        try:
            sol = opti.solve()
            u_opt = sol.value(u) #extract optimal input
            failed = False
        except:
            print("Solver failed!")
            failed = True
            u_opt = np.zeros((self.input_dimn, 1))

        if sol.value(delta_alpha) > 0.01:
            print("delta alpha: ", sol.value(delta_alpha))
        # print("H_curr: ", h)
        #return the optimal input!
        if isinstance(u_opt, float):
            #if input is a float, turn into a 1x1 numpy array
            return np.array([[u_opt]]), h, 0
        return u_opt.reshape((self.input_dimn, 1)), h, 0, failed #return the barrier function and slack variable values as well! 
