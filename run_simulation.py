#Our dependencies
from environment import *
from dynamics import *
from lyapunov_barrier import *
from controller import *
from trajectory import *
from state_estimation import *
from obstacle import *

#system initial condition - give it a small initial x velocity
x0 = np.array([[0, 0, 0, 0.1, 0, 0]]).T

#create a dynamics object for the double integrator
dynamics = DoubleIntDyn(x0)

#create an observer based on the dynamics object with noise parameters
mean = None
sd = None
observer = DoubleIntObserver(dynamics, mean, sd)

#create a circular obstacle
r = 2.5
c = np.array([[5, 5, 0]]).T
circle = Circle(r, c)

#create a depth camera
depthCam = DepthCam(circle, observer, mean = None, sd = None)

#create a depth-based obstacle queue
buffer = 0.1
queueSize = 10
queueType = 'depth'
obstacleQueue = ObstacleQueue(observer, depthCam, buffer, queueSize, queueType=queueType)

#create a trajectory
start = np.array([[0, 0, 0]]).T #Pass in simply spatial dimensions into the system
end = np.array([[10, 10, 0]]).T #goal state in space
T = 10 #Period of trajectory
trajectory = Trajectory(start, end, T)

#create a controller based on the dynamics object
Kp = 1
Kd = 0.5
#Z direction should have no impact on Input.
K = np.array([[Kp, 0, 0, Kd, 0, 0], 
              [0, Kp, 0, 0, Kd, 0], 
              [0, 0, 0, 0, 0, 0]])
controller = StateFB(observer, trajectory)
controller.set_params(K)

#create a simulation environment
env = Environment(dynamics, controller, observer)

#run the simulation
env.run()