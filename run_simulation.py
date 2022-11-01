#Our dependencies
from environment import *
from dynamics import *
from controller import *
from trajectory import *
from state_estimation import *

#system initial condition
x0 = np.zeros((8, 1))

#create a dynamics object for the double integrator
dynamics = QuadDyn(x0)

#create an observer based on the dynamics object with noise parameters
mean = 0
sd = 0.01
observer = QuadObserver(dynamics, mean, sd)

#create a trajectory
start = np.array([[0, 0, 0]]).T #Pass in simply spatial dimensions into the system
end = np.array([[10, 10, 0]]).T #goal state in space
T = 10 #Period of trajectory
trajectory = Trajectory(start, end, T)

#define controller gains
Kp = np.diag([[4, 4, 4]])
Kd = np.diag([[3, 3, 3]])
Ktheta = 1
Komega = 1

#create a planar quadrotor controller
controller = PlanarQrotorPD(observer, lyapunov = None, trajectory = trajectory)
controller.set_params(Kp, Kd, Ktheta, Komega)

#create a simulation environment
env = Environment(dynamics, controller, observer)

#run the simulation
env.run()