#Our dependencies
from environment import *
from dynamics import *
from lyapunov_barrier import *
from controller import *
from trajectory import *
from state_estimation import *
from obstacle import *

#system initial condition
N = 1
if N == 1:
    q0 = np.array([[0, 0, 0.1]]).T
else:
    q0 = np.array([[0, 0, np.pi/4, 0, 0, np.pi/3, 0, 5, -np.pi/4]]).T

#create a dynamics object for the double integrator
dynamics = TurtlebotSysDyn(q0, N = N) #set the number of turtlebots to 1 for now

#create an observer manager based on the dynamics object with noise parameters
mean = 0
sd = 0
observer = ObserverManager(dynamics, mean, sd)

#set a desired state vector for the system
xD = np.array([[1, 5, 0]]).T

#define a trajectory
T = 5
traj = TrajectoryManager(q0, xD, T, N)

#Create a controller manager
controller = ControllerManager(observer, None, traj, 'TurtlebotFBLin')

#create a simulation environment
env = Environment(dynamics, controller, observer)

#run the simulation
env.run()