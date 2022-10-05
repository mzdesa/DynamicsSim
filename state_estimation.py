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
        
    def get_state(self):
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
    
    def get_pos(self):
        """
        Returns a potentially noisy measurement of JUST the position of a double integrator
        Returns:
            3x1 numpy array, observed position vector of systme
        """
        return self.get_state()[0:3].reshape((3, 1))
    
    def get_vel(self):
        """
        Returns a potentially noisy measurement of JUST the velocity of a double integrator
        Returns:
            3x1 numpy array, observed velocity vector of systme
        """
        return self.get_state()[3:].reshape((3, 1))

    def get_orient(self):
        """
        Returns a potentially noisy measurement of JUST the rotation matrix fo a double integrator "car" system.
        Assumes that the system is planar and just rotates about the z axis.
        Returns:
            R (3x3 numpy array): rotation matrix from double integrator "car" frame into base frame
        """
        #first column is the unity vector in direction of velocity. Note that this is already noisy.
        r1 = self.get_vel()
        r1 = r1/np.linalg.norm(r1) #do not recall the get_vel() function as it is stochastic
        #calculate angle of rotation WRT x-axis
        theta = np.arccos(r1[0, 0])
        r2 = np.array([[-np.sin(theta), np.cos(theta), 0]]).T
        #assemble the rotation matrix, normalize the second column, set r3 = e3
        return np.hstack((r1, r2/np.linalg.norm(r2), self.e3))
    
class DepthCam:
    def __init__(self, dynamics, obstacle, observer, stateDimn, inputDimn, mean = None, sd = None):
        """
        Init function for depth camera observer

        Args:
            dynamics (Dynamics): Dynamics object instance
            obstacle (Obstacle): Obstacle object instance
            observer (DoubleIntObserver): Double integrator observer, or one of a similar format
            stateDimn (int): length of state vector
            inputDimn (int): length of input vector
            mean (float, optional): Mean for gaussian noise. Defaults to None.
            sd (float, optional): standard deviation for gaussian noise. Defaults to None.
        """
        self.dynamics = dynamics
        self.obstacle = obstacle
        self.observer = observer
        self.stateDimn = stateDimn
        self.inputDimn = inputDimn
        self.mean = mean
        self.sd = sd
        
        #store the pointcloud dict
        self._ptcloudData = {"ptcloud": None, "rotation": None, "position": None}
        
    def calc_ptcloud(self):
        """
        Calculate the pointcloud over a sample of points
        """
        #define an array of angles
        thetaArr = np.linspace(0, 2*np.pi)
        #get the points in the world frame
        obsPts = self.obstacle.get_pts(thetaArr)
        
        #transform these points into rays in the vehicle frame.
        Rsc = self.observer.get_orient() #get the rotation matrix
        psc = self.observer.get_pos() #get the position
        
        #initialize and fill the pointcloud array
        ptcloud = np.zeros((3, thetaArr.shape[0]))
        for i in range(thetaArr.shape[0]):
            #get the ith XYZ point in the pointcloud (in the spatial frame)
            ps = obsPts[:, i].reshape((3, 1))
            ptcloud[:, i] = (np.linalg.inv(Rsc)@(ps - psc)).reshape((3, ))
            
        #update the pointcloud dict with this data
        self._ptcloudData["ptcloud"] = ptcloud
        self._ptcloudData["rotation"] = Rsc
        self._ptcloudData["position"] = psc
        return self._ptcloudData
        
    def get_pointcloud(self):
        """
        Returns the pointcloud dictionary from the class attribute 
        Returns:
            Dict: dictionary of pointcloud points, rotation matirx, position, and timestamp at capture
        """
        #first, calculate the pointcloud
        self.calc_ptcloud()
        return self._ptcloudData
