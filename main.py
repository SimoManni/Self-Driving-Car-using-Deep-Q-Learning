import pygame
import numpy as np

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

# Function to simulate car using learned policy
def simulate_agent(epoch):
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f'Autonomous Car - Deep Q Learning - Epoch: {epoch}')
    FONT = pygame.font.Font(None, 32)

    score = 0
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


        state = RaceEnv.car.get_state()
        q_values = agent.policy_dqn.predict(state)
        _, reward, done = RaceEnv.step(np.argmax(q_values))
        RaceEnv.draw(screen)

        if done:
            RaceEnv.reset()
            score = 0
        score += reward

        txt_score = FONT.render(f'Score: {score}', True, (0, 0, 0))
        screen.blit(txt_score, (50, 50))

        txt_speed = FONT.render(f'Speed: {RaceEnv.car.speed}', True, (0, 0, 0))
        screen.blit(txt_speed, (50, 70))

        pygame.display.update()
        RaceEnv.draw(screen)

    pygame.quit()

def run():
    for e in range(N_EPISODES):

        if e % 100 == 0:
            simulate_agent(e)

        game.reset()
        score = 0
        counter = 0
        gtime = 0

        state_prev, reward, done = game.step(0)

        while not done:
            action = agent.choose_action(state_prev)
            new_state, reward, done = game.step(action)

            # Countdown if no reward is collected
            if reward == 0:
                counter += 1
                if counter > 100:
                    done = True
            else:
                counter = 0

            score += reward

            agent.remember(state_prev, action, reward, new_state, int(done))
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