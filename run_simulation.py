#Our dependencies
from environment import *
from dynamics import *
from lyapunov_barrier import *
from controller import *
from trajectory import *
from state_estimation import *
from obstacle import *

#system initial condition - give it a small initial x velocity
x0 = np.array([[0, 0, 0, 0.5, 0, 0]]).T

#create a dynamics object for the double integrator
dynamics = DoubleIntDyn(x0)

#create an observer based on the dynamics object with noise parameters
mean = None
sd = None
observer = DoubleIntObserver(dynamics, mean, sd)

#create a circular obstacle
r = 1.5
c = np.array([[5, 5.5, 0]]).T
circle = Circle(r, c)

#create a depth camera
depthCam = DepthCam(circle, observer, mean = None, sd = None)

#create a depth-based obstacle queue
buffer = 1.5+0.3 #set based on radius of obstacle for a static test
queueSize = 1
queueType = 'static' #make it static for now to test out CBF
obstacleQueue = ObstacleQueue(observer, depthCam, buffer, queueSize, queueType=queueType)

if queueType == 'static':
    #set the static queue using the obstacle position
    obstacleQueue.set_static_data([c]) #use the center position defined above

#create a trajectory
start = np.array([[0, 0, 0]]).T #Pass in simply spatial dimensions into the system
end = np.array([[10, 10, 0]]).T #goal state in space
T = 10 #Period of trajectory
trajectory = Trajectory(start, end, T)

#create a reference controller for a CBF QP
Kp = 4
Kd = 3
#Z direction should have no impact on Input.
K = np.array([[Kp, 0, 0, Kd, 0, 0], 
              [0, Kp, 0, 0, Kd, 0], 
              [0, 0, 0, 0, 0, 0]])
refController = StateFB(observer, trajectory)
refController.set_params(K)

#create a CBF QP controller
controller = CBF_QP(observer, trajectory, refController, obstacleQueue)
cbfAlphas = np.array([[1, 5, 6]])
controller.set_params(True, False, cbfAlphas)

#create a simulation environment
env = Environment(dynamics, controller, observer)

#run the simulation
env.run()