# Self-Driving Car using Deep Q-Learning
 This repository contains the implementation of a simple self-driving simulation enviroment in Pygame used to train a Reinforcement Learning Agent using a simple Deep Q-Learning algorithm. The results as well as the game environment can be visualised in the video below. 

<p align="center">
  <img src="https://github.com/SimoManni/Self-Driving-Car-using-Deep-Q-Learning/assets/151052936/252ae138-fd93-473b-a119-69221bdb4df0" alt="RL_Car" width="400">
</p>

## Autonomous Car
The AutonomousCar class encapsulates all essential methods for visualizing the car, simulating its dynamics, and interacting with the environment. This includes handling movement, collision detection, checkpoint tracking, and sensor-based perception to enable reinforcement learning (RL) applications. The main methods are: 

- **State Update (`update`):** 
   - Handles actions such as acceleration, braking, and turning.
   - Applies friction and updates the car's position and orientation.

- **Collision Detection (`check_collision`):**
   - Checks for collisions with the track's contour lines using the car's bounding box by checking if an intersection between lines occured within the contour points of the car. 

- **State and Perception (`get_state` and `perceive`):**
   - Returns the car's state, including distances to perceived obstacles and current speed.

- **Checkpoint Handling (`checkpoint`):**
   - Detects when the car passes checkpoints and updates lap counts.

- **Rendering (`draw`):**
   - Draws the car and its perceived points on the screen.

## RacingEnvironment Class

This class simulates a racing environment for an autonomous car, including track visualization, car movement, collision detection, and checkpoint handling. It includes the definition of a car object and adds additional methods for visualization and simulation. 

- **State update (`step`)**: Updates the car's state based on the action, checks for collisions and checkpoints, calculates rewards, and returns the new state, reward, and whether the episode is done.

- **Rendering (`draw`)**: Renders the environment on the screen, including barriers, checkpoints, and the car.


## Main 
The main file runs the loop for a fixed number of episodes as specified in the 'settings.py' file. At a fixed rate, it also simulates the agent in the environment to see how much progress the car has made in the learning process. 

## Main test
This program loads the learned policy and simulates the autonomous car in the racing environment, as shown in the video above. 
