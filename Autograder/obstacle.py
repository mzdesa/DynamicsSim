import numpy as np
from lyapunov_barrier import PointBarrier

#File containing classes involved in managing obstacle avoidance.

class Circle:
    def __init__(self, r, c):
        """
        Init function for a planar circular obstacle.
        Represents the geometry of the obstacle.
        Args:
            r (float): radius of circular obstacle
            c (3x1 numpy array): center position of obstacle
        """
        self._r = r
        self._c = c
        
    def get_radius(self):
        """
        Return the radius of the obstacle
        """
        return self._r
    
    def get_center(self):
        """
        Return the center of the obstacle in the world frame.
        Note that z coordinate should be 0
        """
        return self._c
    
    def get_pts(self, thetaArr):
        """
        Returns a set of points on the obstacle for a set of angles around the circle
        Args:
            thetaArr ((N, ) numpy array): array of N theta values to evaluate the obstalce position
        Returns:
            (3xN) Numpy array: set of N points on the circle in R3
        """
        obsPts = np.zeros((3, thetaArr.shape[0]))
        for i in range(thetaArr.shape[0]):
            #fill in the column of the obstacle with the appropriate point
            obsPts[:, i] = (np.array([[self._r*np.cos(thetaArr[i]), self._r*np.sin(thetaArr[i]), 0]]).T + self._c).reshape((3, ))
        return obsPts
    
class ObstacleQueue:
    def __init__(self, observer, depthCam, buffer, queueSize, queueType = 'static'):
        """
        Class to keep track of the obstacles through barrier functions.
        Args:
            observer (Observer): observer object
            depthCam (DepthCam): depth camera object
            buffer (float): buffer to be used in CBF computation
            queueSize (float): number of CBFs to be using simultaneously
            queueType (string): 'static' if a constant CBF (applied to entire obs), 'depth' if a depth-based CBF (applied pointwise)
        """
        #store input parameters
        self.observer = observer
        self.depthCam = depthCam
        self.buffer = buffer
        self.queueSize = queueSize
        self.queueType = queueType
        
        #create queue of queueSize PointBarriers
        self.barriers = [PointBarrier(self.observer.inputDimn, self.observer.stateDimn, self.observer.dynamics, observer, buffer) for i in range(queueSize)]
    
    def set_static_data(self, centerPts):
        """
        Function to set the barrier points for static obstacles
        Args:
            centerPts (List): list of queueSize [x, y, z] center points of the obstacle for the barriers to use
        """
        i = 0
        for bar in self.barriers:
            bar.set_barrier_pt(centerPts[i])
            i+=1
    
    def depth_to_spatial(self, ptMatrix):
        """
        Converts a matrix of points from the vehicle frame to the spatial frame.
        Args:
            ptMatrix (stateDimn x QueueSize numpy array): matrix where each column is a point in the vehicle frame
        Returns:
            (stateDimn x QueueSize numpy array): ptMatrix, where each column has been converted into the spatial frame
        """
        #extract some relevant state information
        R_sc = self.observer.get_orient() #rotation matrix from the car frame to the world frame. (3x3) NumPy Array
        carPos = self.observer.get_pos() #position of the car in the world frame, (3x1) NumPy Array
        carVel = self.observer.get_vel() #velocity of the car in the world frame, (3x1) NumPy Array
        
        #*****************************************************************************************
        #YOUR CODE HERE:
        #In this function, we want to figure out how to convert an entire matrix of points
        #in the car frame into the world frame. We pass in ptMatrix, a matrix of depth camera
        #readings in the car frame. Each column in the matrix is one point that's read by the
        #depth camera. We want to figure out how to transform each column of this matrix into the 
        #world frame. There are several approaches to this problem - try to acheieve this using as 
        #few operations as possible to make the simulator run faster. You can even acheive this in 
        #a single line of code! Hint: look at the np.tile() function.
        #*****************************************************************************************
        
        #convert the matrix back into the world frame
        transformedPoints = ptMatrix #PLACEHOLDER VALUE. YOU SHOULD REPLACE THIS
        return transformedPoints
        
    def update_queue(self):
        """
        Updates the barrier queue depending on the motion of the system.
        Will only need to update the queue if the barrier is depth-based.
        """
        if self.queueType == 'depth':
            #solve for latest KNN points in the VEHICLE FRAME - maybe have them implement the KNN function as well?
            knn = self.depthCam.get_knn(self.queueSize)
            
            #convert these points into the spatial frame
            knnSpatial = self.depth_to_spatial(knn)
            #update each barrier point according to the KNN
            for i in range(self.queueSize):
                self.barriers[i].set_barrier_pt(knnSpatial[:, i].reshape((3, 1)))