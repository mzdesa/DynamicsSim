#Our dependencies
from environment import *
from dynamics import *
from lyapunov_barrier import *
from controller import *
from trajectory import *
from state_estimation import *
from obstacle import *
from utils import *

simpleTest = True

#system initial condition
if not simpleTest:
    #for standard case, use 5 turtlebots
    N = 5
    #generate initial condition
    q0 = gen_init_cond(N)
    #generate goal state
    qD = gen_goal_state(N)
else:
    #for simple case, use 2 turtlebots
    N = 2
    #generate initial condition
    q0 = np.array([[5, 2.505, 0, 0, 2.495, 0]]).T
    #generate goal state
    qD = np.array([[0, 2.5, 0, 5, 2.5, 0]]).T

#create a dynamics object for the double integrator
dynamics = TurtlebotSysDyn(q0, N = N) #set the number of turtlebots to 1 for now

#create an observer manager based on the dynamics object with noise parameters
mean = 0
sd = 0
observerManager = ObserverManager(dynamics, mean, sd)

#define a trajectory manager
T = 10
trajManager = TrajectoryManager(q0, qD, T, N)

#define a lidar manager and lidar noise parameters
m = 0
s = 0
lidarManager = LidarManager(observerManager, m, s)

#define a barrier manager
useDeadLock = True
useVision = False
if not useVision:
    barrierManager = BarrierManager(N, 3, 2, dynamics, observerManager, 0, dLock = useDeadLock, lidarManager = None)
else:
    barrierManager = BarrierManager(N, 3, 2, dynamics, observerManager, 0, dLock = useDeadLock, lidarManager = lidarManager)

#Create a controller manager
if useDeadLock and not useVision:
    ctrlType = 'TurtlebotCBFQPDeadlock'
else:
    # ctrlType = 'TurtlebotFBLin'
    ctrlType = 'TurtlebotCBFQPVision'
controller = ControllerManager(observerManager, barrierManager, trajManager, lidarManager, ctrlType)

#create a simulation environment
env = Environment(dynamics, controller, observerManager, T = T)

#run the simulation
env.run()