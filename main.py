import pygame
import numpy as np

from settings import *
from Environment import SimulationEnvironment
from Q_learning import Agent


game = SimulationEnvironment()

MAX_TIME_SECONDS = 5
GameTime = 0
GameHistory = []

agent = Agent()
# agent.load_model()

ddqn_scores = []
eps_history = []

def simulate_agents(epoch):
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f'Autonomous Car - Deep Q Learning - Epoch: {epoch}')

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


        states = game.get_state()
        actions = agent.get_actions(states)
        _, _, done_array = game.step(actions)
        for i, done in enumerate(done_array):
            if done:
                game.cars[i].reset()

        game.draw(screen)
        pygame.display.update()

    # Quit Pygame
    pygame.quit()



def run():
    for e in range(N_EPISODES):

        if e % 10 == 0:
            simulate_agents(e)

        game.reset()
        score = 0
        counter = np.zeros(N_CARS)
        gtime = 0

        state_prev, reward, done = game.step(np.zeros(N_CARS))

        while not np.any(done):
            actions = agent.get_actions(state_prev)
            new_state, reward, done = game.step(actions)

            # Countdown if no reward is collected
            counter[reward == 0] += 1
            if np.any(counter > 100):
                done[counter > 100] = True

            score += np.mean(reward)

            agent.remember(state_prev, actions, reward, new_state, done.astype(int))
            agent.learn()
            state_prev = new_state

            gtime += 1

            if gtime >= TOTAL_GAMETIME:
                done = True

        if e % 10 == 0 and e > 10:
            agent.save_model()
            print("save model")


        eps_history.append(agent.epsilon)
        ddqn_scores.append(score)
        avg_score = np.mean(ddqn_scores[max(0, e - 100):(e + 1)])

        print('Episode: ', e,
              ', Score: %.2f' % score,
              ', Average score %.2f' % avg_score,
              ', Epsilon: ', agent.epsilon,
              ', Memory size', agent.memory.mem_cntr % MEM_SIZE)


run()