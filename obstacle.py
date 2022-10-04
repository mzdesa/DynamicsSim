from winsound import SND_PURGE
import numpy as np

class Obstacle:
    def __init__(self, r, c):
        """
        Init function for a planar circular obstacle
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