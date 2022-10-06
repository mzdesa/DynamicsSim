import numpy as np
from scipy.spatial import cKDTree

class StateObserver:
    def __init__(self, dynamics, mean = None, sd = None):
        """
        Init function for state observer

        Args:
            dynamics (Dynamics): Dynamics object instance
            mean (float, optional): Mean for gaussian noise. Defaults to None.
            sd (float, optional): standard deviation for gaussian noise. Defaults to None.
        """
        self.dynamics = dynamics
        self.stateDimn = dynamics.stateDimn
        self.inputDimn = dynamics.inputDimn
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
    def __init__(self, dynamics, mean, sd):
        """
        Init function for state observer

        Args:
            dynamics (Dynamics): Dynamics object instance
            mean (float, optional): Mean for gaussian noise. Defaults to None.
            sd (float, optional): standard deviation for gaussian noise. Defaults to None.
        """
        super().__init__(dynamics, mean, sd)
        
        #define standard basis vectors to refer to
        self.e1 = np.array([[1, 0, 0]]).T
        self.e2 = np.array([[0, 1, 0]]).T
        self.e3 = np.array([[0, 0, 1]]).T
    
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
        #extract the position and velocity of the vehicle in the world frame
        carPos = self.get_pos() #position of the car in the world frame, (3x1) NumPy array
        carVel = self.get_vel() #velocity of the car in the world frame, (3x1) NumPy array
        
        #**************************************************************************************
        #TODO: YOUR CODE HERE
        #Our car has position and velocity readings, but has no explicit orientation readings.
        #In this question, we want to solve for the Rotation matrix from the car frame into the 
        #world frame using just the position and velocity of the car.
        #**************************************************************************************
        R_sc = np.eye(3) #PLACEHOLDER VALUE. Replace this!
        return R_sc
    
class DepthCam:
    def __init__(self, circle, observer, mean = None, sd = None):
        """
        Init function for depth camera observer

        Args:
            circle (Circle): Circle object instance
            observer (DoubleIntObserver): Double integrator observer, or one of a similar format
            stateDimn (int): length of state vector
            inputDimn (int): length of input vector
            mean (float, optional): Mean for gaussian noise. Defaults to None.
            sd (float, optional): standard deviation for gaussian noise. Defaults to None.
        """
        self.circle = circle
        self.observer = observer
        self.stateDimn = self.observer.stateDimn
        self.inputDimn = self.observer.inputDimn
        self.mean = mean
        self.sd = sd
        
        #stores the pointcloud dict in the vehicle frame (not spatial)
        self._ptcloudData = {"ptcloud": None, "rotation": None, "position": None, "kd": None}
        
    def calc_ptcloud(self):
        """
        Calculate the pointcloud over a sample of points
        """
        #define an array of angles
        thetaArr = np.linspace(0, 2*np.pi)
        #get the points in the world frame
        obsPts = self.circle.get_pts(thetaArr)
        
        #transform these points into rays in the vehicle frame.
        Rsc = self.observer.get_orient() #get the rotation matrix
        psc = self.observer.get_pos() #get the position
        
        #initialize and fill the pointcloud array
        ptcloud = np.zeros((3, thetaArr.shape[0]))
        for i in range(thetaArr.shape[0]):
            #get the ith XYZ point in the pointcloud (in the spatial frame)
            ps = obsPts[:, i].reshape((3, 1))
            ptcloud[:, i] = (np.linalg.inv(Rsc)@(ps - psc)).reshape((3, ))
            
        #generate the KD tree associated with the data
        kdtree = cKDTree(ptcloud.T) #must store in transpose
            
        #update the pointcloud dict with this data
        self._ptcloudData["ptcloud"] = ptcloud
        self._ptcloudData["rotation"] = Rsc
        self._ptcloudData["position"] = psc
        self._ptcloudData["kd"] = kdtree
        return self._ptcloudData
        
    def get_pointcloud(self, update = True):
        """
        Returns the pointcloud dictionary from the class attribute 
        Args:
            update: whether or not to recalculate the pointcloud
        Returns:
            Dict: dictionary of pointcloud points, rotation matrix, position, and timestamp at capture
        """
        #first, calculate the pointcloud
        if update:
            self.calc_ptcloud()
        return self._ptcloudData
    
    def get_knn_scipy(self, K):
        """
        Gets the K Nearest neighbors in the pointcloud to the point. Uses the SCIPY cKD tree.
        Args:
            K (int): number of points to search for
        Returns:
            (3xK numpy array): Matrix of closest points in the vehicle frame
        """
        #check what's closest to the zero vector - this will give the closest points in the vehicle frame!
        dist, ind = self.get_pointcloud(update = True)["kd"].query(np.zeros((1, 3)), K)
        
        #extract list
        if ind.shape != (1, ):
            ind = ind[0]
        
        #convert indices to a matrix of the points
        closest_K = np.zeros((3, K))
        for i in range(K):
            index = ind[i] #extract the ith index from the optimal index list
            closest_K[:, i] = (self._ptcloudData["ptcloud"])[:, index]
        
        #return the matrix
        return closest_K
    
    def get_knn(self, K):
        """
        Gets the K Nearest neighbors in the pointcloud to the point. Uses a numpy implementation, written by you :)
        Args:
            K (int): number of points in the pointcloud to search for. Assume K <= Number of points in the pointcloud.
        Returns:
            (3xK numpy array): Matrix of closest points in the VEHICLE frame. Each column is a point.
        """
        #retrieve the pointcloud data
        #ptcloud will be a 3xN numpy array, containing points in space sensed by the depth camera
        ptcloud = self._ptcloudData["ptcloud"]
        
        #retrieve the current state of the vehicle in the vehicle frame
        xVehicle = np.zeros((3, 1)) #Since the pointcloud is in the vehicle frame, we use (0, 0, 0) to represent the position of the vehicle.
        
        #*************************************************************************************
        #TODO: YOUR CODE HERE: 
        #WRITE A FUNCTION TO SEARCH FOR THE K NEAREST POINTS IN THE POINTCLOUD
        #Your function must be written from scratch using numpy. Note that there are no constraints
        #on how fast your code should be. As long as it works, you're all good to go, although
        #we encourage you to make it as efficient as possible to get the most out of this exercise.
        #The two variables provided above are the pointcloud of the vehicle and the position of the
        #vehicle in the vehicle frame (this is the zero vector). Since the depth
        #camera is attached to the vehicle, this pointcloud is in the vehicle frame.
        #*************************************************************************************
        closest_K = self.get_knn_scipy(K) #Placeholder value to ensure the code runs. You should put your implementation here!
        return closest_K
