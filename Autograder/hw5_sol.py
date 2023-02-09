#Contains implementation of the three required functions for hw 5
#Note: passes in class parameters as function arguments
import numpy as np


#SOLUTION FOR KNN - REFER TO THE SCIPY VERSION
def get_knn_soln(K, ptcloud, ptcloud_KD):
    """
    Gets the K Nearest neighbors in the pointcloud to the point and their indices.
    Args:
        K (int): number of points to search for
        ptcloud: Numpy pointcloud arrray
        ptcloud_KD: scipy KD tree corresponding to pointcloud array
    Returns:
        (3xK numpy array): Matrix of closest points in the vehicle frame
    """
    #check what's closest to the zero vector - this will give the closest points in the vehicle frame!
    dist, ind = ptcloud_KD.query(np.zeros((1, 3)), K)
    
    #extract list
    if ind.shape != (1, ):
        ind = ind[0]
    
    #convert indices to a matrix of the points
    closest_K = np.zeros((3, K))
    for i in range(K):
        index = ind[i] #extract the ith index from the optimal index list
        closest_K[:, i] = ptcloud[:, index]
    
    #return the matrix
    return closest_K

#SOLUTION FOR GETTING ORIENTATION
def get_orient_soln(carVel):
    """
    Returns a potentially noisy measurement of JUST the rotation matrix fo a double integrator "car" system.
    Assumes that the system is planar and just rotates about the z axis.
    Inputs:
        carVel (3x1 numpy array): velocity of car in the spatial frame
    
    Returns:
        R (3x3 numpy array): rotation matrix from double integrator "car" frame into base frame
    """
    #first column is the unity vector in direction of velocity. Note that this is already noisy.
    r1 = carVel #self.get_vel()
    if(np.linalg.norm(r1) != 0):
        #do not re-call the get_vel() function as it is stochastic
        r1 = r1/np.linalg.norm(r1) 
    else:
        #set the r1 direction to be e1 if velocity is zero
        r1 = np.array([[1, 0, 0]]).T
        
    #calculate angle of rotation WRT x-axis
    theta = np.arccos(r1[0, 0])
    r2 = np.array([[-np.sin(theta), np.cos(theta), 0]]).T
    
    #assemble the rotation matrix, normalize the second column, set r3 = e3
    return np.hstack((r1, r2/np.linalg.norm(r2), np.array([[0, 0, 1]]).T))

#SOLUTION FOR DEPTH TO SPATIAL
def depth_to_spatial_soln(ptMatrix, R, carPos):
    """
    Converts a matrix of points from the vehicle frame to the spatial frame.
    Args:
        ptMatrix (stateDimn x QueueSize numpy array): matrix where each column is a point in the vehicle frame
        R: rotation matrix from car to world frame (self.observer.get_orient())
        carPos: position of car (self.observer.get_pos())
    Returns:
        (stateDimn x QueueSize numpy array): ptMatrix, where each column has been converted into the spatial frame
    """
    #convert the matrix back into the world frame
    return R@ptMatrix + np.tile(carPos, (1, ptMatrix.shape[1]))