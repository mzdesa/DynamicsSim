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
        """
        obsPts = np.zeros((3, thetaArr.shape[0]))
        for i in range(thetaArr.shape[0]):
            #fill in the column of the obstacle with the appropriate point
            obsPts[:, i] = (np.array([[self._r*np.cos(thetaArr[i]), self._r*np.sin(thetaArr[i]), 0]]).T + self._c).reshape((3, ))
        return obsPts
    
class BarrierQueue:
    def __init__(self, dynamics, observer, depthCam, buffer, queueSize):
        """
        Class to keep track of the barrier functions used to avoid obstacles.
        Args:
            dynamics (Dynamics): dynamics object
            observer (Observer): observer object
            depthCam (DepthCam): depth camera object
            buffer (float): buffer to be used in CBF computation
            queueSize (float): number of CBFs to be using simultaneously
        """
        #store input parameters
        self.dynamics = dynamics
        self.observer = observer
        self.depthCam = depthCam
        self.buffer = buffer
        self.queueSize = queueSize
        
        #create queue of queueSize PointBarriers
        self.barriers = [PointBarrier(dynamics.inputDimn, dynamics.stateDimn, dynamics, observer, buffer) for i in range(queueSize)]
        
    def depth_to_spatial(self, ptMatrix):
        """
        NOTE: THIS IS WHAT STUDENTS SHOULD IMPLEMENT
        Converts a matrix of points from the vehicle frame to the spatial frame.
        Args:
            ptMatrix (stateDimn x QueueSize numpy array): matrix where each column is a point in the vehicle frame
        Returns:
            (stateDimn x QueueSize numpy array): ptMatrix, where each column has been converted into the spatial frame
        """
        #convert the matrix back into the world frame
        return self.observer.get_orient()@ptMatrix + np.tile(self.observer.get_pos(), (1, self.queueSize))
        
    def update_queue(self):
        """
        Updates the barrier queue depending on the motion of the system.
        """
        #solve for latest KNN points in the VEHICLE FRAME - maybe have them implement the KNN function as well?
        knn = self.depthCam.get_knn(self.queueSize)
        
        #convert these points into the spatial frame
        knnSpatial = self.depth_to_spatial(knn)
        
        #update each barrier point according to the KNN
        for i in range(self.queueSize):
            self.barriers[i].set_barrier_pt(knnSpatial[:, i].reshape((3, 1)))