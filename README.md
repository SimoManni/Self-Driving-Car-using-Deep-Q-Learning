# Self-Driving Car using Deep Q-Learning
 This repository contains the implementation of a simple self-driving simulation enviroment in Pygame used to train a Reinforcement Learning Agent using a simple Deep Q-Learning algorithm. The results as well as the game environment can be visualised in the video below. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/a507e87f-fc9a-4f89-9cdd-5b76bad8ec1a" alt="RL_Car" width="500">
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

In addition, the constructor takes a boolean variable as input, which is used for the definition of the starting configuration. When `random = True`, the car randomly picks a starting position and corresponding angle; furthermore, the checkpoints are sorted in such a way that the car gets rewards only when going forward along the track. This configuration allows for the creation of multiple cars during the learning phase. On the other hand, when the class is used to test the discovered policy, `random = False` and the starting configuration and checkpoints are defined in a way that can be manipulated by the user.


## Environment 
The `Environment` file contains two classes that both simulate a racing environment for an autonomous car, including track visualization, car movement, collision detection, and checkpoint handling. The difference, is that the **`SimulationEnvironment`** handles multiple cars, while the **`RacingEnvironment`** class simulates just one car with a policy. The main methods of the two classes are:

- **State update (`step`)**: Updates the car's state based on the action, checks for collisions and checkpoints, calculates rewards, and returns the new state, reward, and whether the episode is done.

- **Rendering (`draw`)**: Renders the environment on the screen, including barriers, checkpoints, and the car.


## Main 
The main file runs the loop for a fixed number of episodes as specified in the 'settings.py' file. At a fixed rate, it also simulates the agent in the environment to see how much progress the car has made in the learning process. As anticipated earlier, it was foud through empirical analysis, that when I single car was used, the results generalized poorly to different starting configurations than the one used during training. To tackle this problem, a solution was to run multiple cars in parallel starting from multiple initial configurations, so that the agent could observe different states and hopefully learn a policy that better adapts to never-before seen states. 

One example of the visualization of the learning simulation can be visualized in the following video: 
<p align="center">
  <img src="https://github.com/user-attachments/assets/9812c99b-a132-42f7-8aca-ce9360e018c9" alt="RL_Car" width="500">
</p>

## Main test
This program loads the learned policy and simulates the autonomous car in the racing environment, as shown in the video above. The starting configuration can be chosen in the `settings.py` file. 

### Results and Further Improvements
The solution to simulate multiple cars at the same time to show the agent states that it wouldn't have seen otherwise, greatly improved the generalization capabilities of the learned policy, allowing the agent to drive correctly the car around the track from multiple different configurations. However, the policy still has problems sometimes when starting in a curve, as it tries to steer the car when the velocity is null, probably because this is how the agent has learned to take turns, but can't figure out to accelerate and turn. One way to solve this would be to start a dedidicated learning on a curve and run it for multiple episodes to allow the agent to learn this diffucult manuever. Overall, though the policy works well and the agent is able to safely drive the car in most situations. 

One further improvement in the future could be the addition of the minimization of the lap time to force the agent to learn to drive the car as fast as possible. This could lead to different results, as the agent might learn to take different racing lines in order to go faster around the track. 
