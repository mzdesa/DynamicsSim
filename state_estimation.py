import numpy as np

class StateObserver:
    def __init__(self, dynamics, stateDimn, inputDimn, mean = None, sd = None):
        """
        Init function for state observer

        Args:
            dynamics (Dynamics): Dynamics object instance
            stateDimn (int): length of state vector
            inputDimn (int): length of input vector
            mean (float, optional): Mean for gaussian noise. Defaults to None.
            sd (float, optional): standard deviation for gaussian noise. Defaults to None.
        """
        self.dynamics = dynamics
        self.stateDimn = stateDimn
        self.inputDimn = inputDimn
        self.mean = mean
        self.sd = sd
        
    def observe(self):
        """
        Returns a potentially noisy observation of the system state
        """
        if self.mean and self.sd:
            #return an observation of the vector with noise
            return self.dynamics.get_state() + np.random.normal(self.mean, self.sd, (self.stateDimn, 1))
        return self.dynamics.get_state()
    
class DoubleIntObserver(StateObserver):
    def __init__(self, dynamics, stateDimn, inputDimn, mean, sd):
        """
        Init function for state observer

        Args:
            dynamics (Dynamics): Dynamics object instance
            stateDimn (int): length of state vector
            inputDimn (int): length of input vector
            mean (float, optional): Mean for gaussian noise. Defaults to None.
            sd (float, optional): standard deviation for gaussian noise. Defaults to None.
        """
        super().__init__(dynamics, stateDimn, inputDimn, mean, sd)
        
        #define standard basis vectors to refer to
        self.e1 = np.array([[1, 0, 0]]).T
        self.e2 = np.array([[0, 1, 0]]).T
        self.e3 = np.array(([0, 0, 1])).T
    
    def observe_pos(self):
        """
        Returns a potentially noisy measurement of JUST the position of a double integrator
        Returns:
            3x1 numpy array, observed position vector of systme
        """
        return self.observe()[0:3].reshape((3, 1))
    
    def observe_vel(self):
        """
        Returns a potentially noisy measurement of JUST the velocity of a double integrator
        Returns:
            3x1 numpy array, observed velocity vector of systme
        """
        return self.observe()[3:].reshape((3, 1))

    def observe_orient(self):
        """
        Returns a potentially noisy measurement of JUST the rotation matrix fo a double integrator "car" system.
        Assumes that the system is planar and just rotates about the z axis.
        Returns:
            R (3x3 numpy array): rotation matrix from double integrator "car" frame into base frame
        """
        #first column is the unity vector in direction of velocity.
        r1 = self.observe_vel()/np.linalg.norm(self.observe_vel())
        #calculate angle of rotation WRT x-axis
        theta = np.arccos(r1[0, 0])
        #use angle of rotation to calculate r2
        r2 = np.array([[-np.sin(theta), np.cos(theta), 0]]).T
        #assemble the rotation matrix (r3 = e3)
        return np.hstack((r1, r2, self.e3))
    
class DepthCam:
    def __init__(self, dynamics, obstacle, stateDimn, inputDimn, mean = None, sd = None):
        """
        Init function for depth camera observer

        Args:
            dynamics (Dynamics): Dynamics object instance
            obstacle (Obstacle): Obstacle object instance
            stateDimn (int): length of state vector
            inputDimn (int): length of input vector
            mean (float, optional): Mean for gaussian noise. Defaults to None.
            sd (float, optional): standard deviation for gaussian noise. Defaults to None.
        """
        self.dynamics = dynamics
        self.obstacle = obstacle
        self.stateDimn = stateDimn
        self.inputDimn = inputDimn
        self.mean = mean
        self.sd = sd
        
        #store the pointcloud dict
        self._ptcloudData = {"ptcloud": None, "rotation": None, "position": None, "time": None}
        
        #store the obstacle points (Private var)
        self._obs_world = None
        
    def calc_ptcloud(self):
        """
        
        """
        
    def set_obs_world(self):
        """
        Defines and returns the position of the obstacle points in the world frame
        """
        #define set of angles to evaluate obstalce position
        thetaArr = np.linspace(0, 2*np.pi)
        self._obs_world = self.obstacle.get_pts(thetaArr)
        return self._obs_world
        
    def get_pointcloud(self):
        """
        Returns the pointcloud dictionary from the class attribute 
        Returns:
            Dict: dictionary of pointcloud points, rotation matirx, position, and timestamp at capture
        """
        return self._ptcloudData
