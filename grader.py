#!/usr/bin/env python3

import json, os
import numpy as np

indiv_scores = 100/2

message = ""

get_orient_score = 0
get_orient_message = ""

depth_to_spatial_score = 0
depth_to_spatial_message = ""

try:
    #Import all necessary files
    from controller import *
    from dynamics import *
    from environment import *
    from lyapunov_barrier import *
    from obstacle import *
    from run_simulation import *
    from state_estimation import *
    from trajectory import *
    from hw5_sol import *

    """
    ****************************************
    RUN SIMULATION SETUP CODE BEFORE TESTING
    ****************************************
    """
    #system initial condition - give it a small initial x velocity
    x0 = np.array([[0, 0, 0, 0.5, 0, 0]]).T

    #create a dynamics object for the double integrator
    dynamics = DoubleIntDyn(x0)

    #create an observer based on the dynamics object with ZERO noise to grade
    mean = None
    sd = None
    observer = DoubleIntObserver(dynamics, mean, sd)

    #create a circular obstacle
    r = 1.5
    c = np.array([[5, 6, 0]]).T
    circle = Circle(r, c)

    #create a depth camera
    depthCam = DepthCam(circle, observer, mean = None, sd = None)

    #create a depth-based obstacle queue
    queueType = 'depth'
    if queueType == 'static':
        #set the static queue using the obstacle position
        buffer = 1.5 + 0.3
        queueSize = 1
        obstacleQueue = ObstacleQueue(observer, depthCam, buffer, queueSize, queueType=queueType)
        obstacleQueue.set_static_data([c]) #use the center position defined above
    else:
        buffer = 0.3 #set based on radius of obstacle for a static test
        queueSize = 10
        obstacleQueue = ObstacleQueue(observer, depthCam, buffer, queueSize, queueType=queueType)
        
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

    #create a simulation environment - Note: don't need to actually run the simulation!
    env = Environment(dynamics, controller, observer)
    
    #First, test out their get_orientat function!
    passed = True
    try:
        #manually set the velocity of the vehicle to a few choices, see what matrix they get.
        vel1 = np.array([[1, 0, 0]]).T
        dynamics._x = vel1 #MANUALLY overrride dynamics.
        
        #check the solutions's rotation matrix.
        RSoln = get_orient_soln(vel1)
        
        #check their rotation matrix.
        RStudent = observer.get_orient()
        
        #check if the two are within tolerance.
        passed = passed and np.allclose(RSoln, RStudent, rtol=1e-2)
        
        if passed:
            get_orient_score = indiv_scores
            get_orient_message = "Correct implementation of get_orient."
        else:
            get_orient_score = 0
            get_orient_message = "Incorrect implementation of get_orient."
    except Exception as e:
        message = f"Unable to run get_orient test. Following exception thrown: \n{e}"
        
    #Next, test out their depth_to_spatial solution
    env.reset() #RESET THE ENVIRONMENT.
    passed = True
    try:
        #create a random ptMatrix array. Remember, queueSize = 10.
        ptMatrix = np.random.uniform(0, 1, (3, queueSize)) #sample a random matrix uniformly on 0 to 1.
        
        #get the rotation matrix using the solution code
        R = get_orient_soln(observer.get_vel())
        
        #get car position
        carPos = observer.get_pos()
        
        #check the solutions's rotation matrix.
        ptMatrixSoln = depth_to_spatial_soln(ptMatrix, R, carPos)
        
        #check their rotation matrix.
        ptMatrixStudent = obstacleQueue.depth_to_spatial(ptMatrix)
        
        #check if the two are within tolerance.
        passed = passed and np.allclose(ptMatrixSoln, ptMatrixStudent, rtol=1e-2)
        
        if passed:
            depth_to_spatial_score = indiv_scores
            depth_to_spatial_message = "Correct implementation of depth_to_spatial."
        else:
            depth_to_spatial_score = 0
            depth_to_spatial_message = "Incorrect implementation of depth_to_spatial."
    except Exception as e:
        message = f"Unable to run depth_to_spatial test. Following exception thrown: \n{e}"

except Exception as e:
    message = f"Unable to run simulation tests. Following exception thrown: \n{e}"


total_score = get_orient_score + depth_to_spatial_score

if total_score >= 99.9:
    message = "WHOOO ALL TEST CASES PASSED 🥳🥳🥳"
else:
    if message:
        message += "\n"
    message += "Aw darn! I'm sure you'll get it soon 👍"

output = {
        "score": total_score,
        "output": message,
        "visibility": "visible",
        "stdout_visibility": "visible",
        "tests": [
            {
                "score": get_orient_score,
                "max_score": indiv_scores,
                "name": "get_orient test",
                "output": get_orient_message,
                "visibility": "visible"
            },
            {
                "score": depth_to_spatial_score,
                "max_score": indiv_scores,
                "name": "depth_to_spatial test",
                "output": depth_to_spatial_message,
                "visibility": "visible"
            }
        ]
        }

with open('results.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=4)