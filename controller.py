import numpy as np
import casadi as ca
from collections import deque

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

class TurtlebotMPC:
    def __init__(self, observer, obstacles, x_d, N = 10):
        """
        Class for a turtlebot model predictive controller
        Args:
            observer (Observer): state observer object
            obstacles (List of Circle Objects): Python list of circular obstacle objects
            x_d ((3x1) NumPy Array): Desired state of the system
            N (Int): Optimization horizon
        """
        #store input parameters
        self.observer = observer
        self.obstacles = obstacles
        self.x_d = x_d

        #store CASADI PARAMETERS
        self.opti = None #instance of ca.Opti()
        self.X = None #optimization variables - initialize as none
        self.U = None
        self.cost = None #optimization cost - initialize as an none
        
        #store solution
        self.opti_sol = None

        #store optimization horizon
        self.N = N

        #store discretization time step
        self.dt = 1/50

    def get_input(self):
        """
        Retrieves input stored in class parameter
        Returns:
            self._u: most recent input stored in class paramter
        """
        return self._u

    def initialize_variables(self):
        """
        Initialize the variables for the optimization
        X Should be a matrix of dimension (3 x N+1). Each column represents one state in the state sequence.
        U Should be a matrix of dimension (2 x N). Each column represents one input in the input sequence.
        """
        self.X = self.opti.variable(3, self.N + 1)
        self.U = self.opti.variable(2, self.N)

    def add_input_constraint(self):
        """
        Define input constraints on the velocity, v, and angular velocity, omega, inputs to the system!
        Recall that each entry of the input, u_i = [v, omega].
        The v and omega inputs should be between [-10 and 10].
        If you add a constraint on two elements in Casadi, it will apply the constraint element-wise!
        """
        for i in range(self.N):
            self.opti.subject_to(self.U[:, i] <= np.ones((2, 1))*10)
            self.opti.subject_to(self.U[:, i] >= np.ones((2, 1))*(-10))

        

    def add_obs_constraint(self):
        """
        Add constraints on the path of the system so that it does not collide with the obstacles.
        self.obstacles contains a list of obstacles! The radius and center of each obstacle may be accessed with:
        obs.get_radius() -> returns the scalar radius of the obstacle
        obs.get_center() -> returns 2x1 NumPy array of center (x, y) position of the obstacle.
        """
        for obs in self.obstacles:
            for i in range(self.N + 1):
                center = obs.get_center()
                radius = obs.get_radius()
                #slice out the (x, y) of the ith column and apply the obstacle constraint
                self.opti.subject_to((self.X[0:2, i] - center).T@(self.X[0:2, i] - center) > radius**2)

    def discrete_dyn(self, q, u):
        """
        Discretized dynamics of the turtlebot.
        Inputs: q(k) = [x, y, phi] (Casadi vector)
                u(k) = [v, omega] (Casadi vector)
        Returns:
                q(k+1) = q(k) + dt*f(q(k), u(k)) - state vector at the next discretized time step (casadi vector)
        """
        return q + ca.vertcat(u[0]*ca.cos(q[2]), u[0]*ca.sin(q[2]), u[1])*self.dt

    def add_dyn_constraint(self):
        """
        Enforce the dynamics constraint, xk+1 = f(xk, uk), on each element of the system.
        Enforce the initial condition of the optimization as well.
        """
        #add initial condition constraint
        self.opti.subject_to(self.X[:, 0] == self.observer.get_state())

        #add dynamics constraint
        for i in range(self.N-1):
            self.opti.subject_to(self.X[:, i+1] == self.discrete_dyn(self.X[:, i], self.U[:, i]))

    def add_cost(self):
        """
        Compute the cost for the MPC problem.
        """
        #First, define P, Q, R matrices
        P = np.diag((2, 2, 0))
        Q = np.diag((2, 2, 0))
        R = np.eye(2)

        termCost = (self.X[:, self.N] - self.x_d).T@P@(self.X[:, self.N] - - self.x_d)
        sumCost = 0
        for i in range(self.N):
            sumCost = sumCost + (self.X[:, i] - self.x_d).T@Q@(self.X[:, i] - self.x_d) + (self.U[:, i]).T@R@(self.U[:, i])

        self.cost = termCost + sumCost
        # self.cost = sumCost


    def add_warm_start(self):
        """
        Warm starts the system with a guess of the Geometric PD controller
        """
        x0 = ((self.observer.get_state())[0:2]).reshape((2, 1))
        x0 = np.vstack((x0, 0))
        xG = np.linspace(x0[0, 0], self.x_d[0, 0], self.N+1)
        yG = np.linspace(x0[1, 0], self.x_d[1, 0], self.N+1)
        pG = np.linspace(x0[2, 0], self.x_d[2, 0], self.N+1)
        xGuess = ca.DM(np.vstack((xG, yG, pG))) #linearly interpolate between them
        self.opti.set_initial(self.X, xGuess)
        
    def setup(self):
        """
        Setup optimization problem for the first time
        Inputs:
        quad_state: current state of quadrotor, [pos, vel, rot, ang_vel]
        des_state: desired state of quadrotor
        des_accel: desired acceleration of quadrotor (by default 0)
        """        
        #Define optimization problem
        self.opti = ca.Opti()
        
        #Define optimization variables
        self.initialize_variables()
        
        #Define constraint on/off switches
        input_constr = False #Turn on input constraints
        obs_constr = True #Turn on barrier constraints
        
        #Add constraints
        if input_constr:
            self.add_input_constraint()
        if obs_constr:
            self.add_obs_constraint()

        #Add obstacle constraint
        self.add_dyn_constraint()
            
        #Compute cost
        self.add_cost()
        
        #Add warm start linear interpolation guess
        self.add_warm_start()

    def solve_nlp(self):
        """
        Solve the optimization problem
        """
        #Solve optimization
        self.opti.minimize(self.cost)
        option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
        self.opti.solver("ipopt", option)
        self.opti_sol = self.opti.solve()
        #Extract the optimal u vector
        u_opti = self.opti_sol.value(self.U)
        x_opti = self.opti_sol.value(self.X)
        solCost = self.opti_sol.value(self.cost)
        print(solCost)
        return u_opti, x_opti

    def eval_input(self, t):
        """
        Solve for and return control input
        Inputs:
            t (float): time in simulation
        Returns:
            u ((Dynamics.inputDimn x 1)): input vector, as determined by controller
        """
        #set up the optimization at this step
        self.setup()
            
        #using the updated state, solve the problem
        u_opti, x_opti = self.solve_nlp()
                                
        #store the first input in the sequence in the self._u parameter
        self._u = (u_opti[:, 0]).reshape((2, 1))

        #return the full force vector
        return self._u