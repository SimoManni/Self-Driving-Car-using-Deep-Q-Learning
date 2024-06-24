import pygame
import numpy as np
import sys

from settings import *
from RacingEnvironment import RacingEnvironment
from Q_learning import Agent


game = RacingEnvironment()

MAX_TIME_SECONDS = 5
GameTime = 0
GameHistory = []

agent = Agent()

ddqn_scores = []
eps_history = []

def simulate_agent(epoch):
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f'Autonomous Car - Deep Q Learning - Epoch: {epoch}')

    RaceEnv = RacingEnvironment()

    running = True
    start_time = pygame.time.get_ticks()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Check elapsed time
        current_time = pygame.time.get_ticks()  # Get current time in milliseconds
        elapsed_time_seconds = (current_time - start_time) / 1000.0  # Convert to seconds

        if elapsed_time_seconds >= MAX_TIME_SECONDS:
            running = False  # Exit the loop if maximum time exceeded


        observation = RaceEnv.car.perceive()
        action = agent.brain_target(observation).detach().numpy()
        RaceEnv.car.update(np.argmax(action))
        RaceEnv.draw(screen)

    # Quit Pygame
    pygame.quit()

def run():
    for e in range(N_EPISODES):

        if e == 0 or e % 10 == 0:
            simulate_agent(e)

        game.reset()  # reset env

        done = False
        score = 0
        counter = 0

        state_prev, reward, done = game.step(0)

        gtime = 0  # set game time back to 0

        while not done:
            action = agent.choose_action(state_prev)
            observation, reward, done = game.step(action)

            # This is a countdown if no reward is collected the car will be done within 100 ticks
            if reward == 0:
                counter += 1
                if counter > 100:
                    done = True
            else:
                counter = 0

            score += reward

            agent.remember(state_prev, action, reward, observation, int(done))
            agent.learn()

            gtime += 1

            if gtime >= TOTAL_GAMETIME:
                done = True


        eps_history.append(agent.epsilon)
        ddqn_scores.append(score)
        avg_score = np.mean(ddqn_scores[max(0, e - 100):(e + 1)])

        if e % REPLACE_TARGET == 0 and e > REPLACE_TARGET:
            agent.update_network_parameters()

        if e % 10 == 0 and e > 10:
            agent.save_model()
            print("save model")

        print('Episode: ', e,
              ', Score: %.2f' % score,
              ', Average score %.2f' % avg_score,
              ', Epsilon: ', agent.epsilon,
              ', Memory size', agent.memory.mem_cntr % MEM_SIZE)


run()