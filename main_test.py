import pygame
import numpy as np

from settings import *
from RacingEnvironment import RacingEnvironment
from Q_learning import Agent


agent = Agent()
agent.load_model()


# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(f'Autonomous Car - Deep Q Learning')
FONT = pygame.font.Font(None, 32)

score = 0
RaceEnv = RacingEnvironment()

running = True

start_time = pygame.time.get_ticks()
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    state = RaceEnv.car.get_state()
    q_values = agent.policy_dqn.predict(state)
    _, reward, done = RaceEnv.step(np.argmax(q_values))

    if done:
        RaceEnv.reset()
        score = 0
    score += reward

    txt_score = FONT.render(f'Score: {score}', True, (0, 0, 0))
    screen.blit(txt_score, (50, 50))

    txt_speed = FONT.render(f'Speed: {RaceEnv.car.speed:.1f}', True, (0, 0, 0))
    screen.blit(txt_speed, (50, 70))

    pygame.display.update()
    RaceEnv.draw(screen)


pygame.quit()