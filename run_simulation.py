#Our dependencies
from environment import *
from dynamics import *
from lyapunov_barrier import *
from controller import *
from trajectory import *
from state_estimation import *
from obstacle import *

#system initial condition
N = 3
if N == 1:
    q0 = np.array([[0, 0, 0.1]]).T
else:
    q0 = np.array([[0, 0, 0, 5, 0, 0, 0, 5, 0]]).T

#create a dynamics object for the double integrator
dynamics = TurtlebotSysDyn(q0, N = N) #set the number of turtlebots to 1 for now

#create an observer manager based on the dynamics object with noise parameters
mean = 0
sd = 0
observerManager = ObserverManager(dynamics, mean, sd)

#set a desired state vector for the system
xD = np.array([[5, 5, 0, 0, 5, 0, 5, 0, 0]]).T

#define a trajectory manager
T = 5
trajManager = TrajectoryManager(q0, xD, T, N)

#define a barrier manager
barrierManager = BarrierManager(N, 3, 2, dynamics, observerManager, 0)

#Create a controller manager
controller = ControllerManager(observerManager, barrierManager, trajManager, 'TurtlebotCBFQP')

#create a simulation environment
env = Environment(dynamics, controller, observerManager)

#run the simulation
env.run()