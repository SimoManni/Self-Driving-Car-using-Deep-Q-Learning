import pygame

from settings import *
from Environment import RacingEnvironment


# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(f'Autonomous Car - Deep Q Learning')
FONT = pygame.font.Font(None, 32)

score = 0
RaceEnv = RacingEnvironment(screen)

running = True

start_time = pygame.time.get_ticks()
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    state = RaceEnv.get_state()
    action = RaceEnv.get_action(state)
    print(f'Action: {action}')
    reward, done = RaceEnv.update(action)

    if done:
        RaceEnv.reset()
        score = 0
    score += reward

    RaceEnv.draw()
    RaceEnv.write(score)
    pygame.display.update()



pygame.quit()