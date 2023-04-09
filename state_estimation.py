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
        if self.mean or self.sd:
            #return an observation of the vector with noise
            return self.dynamics.get_state() + np.random.normal(self.mean, self.sd, (self.stateDimn, 1))
        return self.dynamics.get_state()
    
class EgoTurtlebotObserver(StateObserver):
    def __init__(self, dynamics, mean, sd, index):
        """
        Init function for a state observer for a single turtlebot within a system of N turtlebots
        Args:
            dynamics (Dynamics): Dynamics object for the entire turtlebot system
            mean (float): Mean for gaussian noise. Defaults to None.
            sd (float): standard deviation for gaussian noise. Defaults to None.
            index (Integer): index of the turtlebot in the system
        """
        #initialize the super class
        super().__init__(dynamics, mean, sd)

        #store the index of the turtlebot
        self.index = index
    
    def get_state(self):
        """
        Returns a potentially noisy measurement of the state vector of the ith turtlebot
        Returns:
            3x1 numpy array, observed state vector of the ith turtlebot in the system (zero indexed)
        """
        # print('hi')
        return super().get_state()[3*self.index : 3*self.index + 3].reshape((3, 1))
    
    def get_vel(self):
        """
        Returns a potentially noisy measurement of the derivative of the state vector of the ith turtlebot
        Inputs:
            i (int): the index of the desired turtlebot position in the system (zero indexed)
        Returns:
            3x1 numpy array, observed derivative of the state vector of the ith turtlebot in the system (zero indexed)
        """
        #first, get the current input to the system of turtlebots
        u = self.dynamics.get_input()

        #now, get the noisy measurement of the entire state vector
        x = self.get_state()

        #to pass into the deriv function, augment x with zeros elsewhere
        x = np.vstack((np.zeros((self.index*3, 1)), x, np.zeros(((self.dynamics.N - 1 - self.index)*3, 1))))
        
        #calculate the derivative of the ith state vector using the noisy state measurement
        xDot = self.dynamics.deriv(x, u, 0) #pass in zero for the time (placeholder for time invar system)

        #slice out the derivative of the ith turtlebot and reshape
        return xDot[3*self.index : 3*self.index + 3].reshape((3, 1))
    
    
class ObserverManager:
    def __init__(self, dynamics, mean, sd):
        """
        Managerial class to manage the observers for a system of N turtlebots
        Args:
            dynamics (Dynamics): Dynamics object instance
            mean (float, optional): Mean for gaussian noise. Defaults to None.
            sd (float, optional): standard deviation for gaussian noise. Defaults to None.
        """
        #store the input parameters
        self.dynamics = dynamics
        self.mean = mean
        self.sd = sd

        #create an observer dictionary storing N observer instances
        self.observerDict = {}

        #create N observer objects
        for i in range(self.dynamics.N):
            #create an observer with index i
            self.observerDict[i] = EgoTurtlebotObserver(dynamics, mean, sd, i)

    def get_observer_i(self, i):
        """
        Function to retrieve the ith observer object for the turtlebot
        Inputs:
            i (integet): index of the turtlebot whose observer we'd like to retrieve
        """
        return self.observerDict[i]
    
    def get_state(self):
        """
        Returns a potentially noisy observation of the *entire* system state (vector for all N bots)
        """
        #get each individual observer state
        xHatList = []
        for i in range(self.dynamics.N):
            #call get state from the ith observer
            xHatList.append(self.get_observer_i(i).get_state())

        #vstack the individual observer states
        return np.vstack(xHatList)

    
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
    
    def get_knn(self, K):
        """
        Gets the K Nearest neighbors in the pointcloud to the point and their indices.
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
