import pygame
from RacingEnvironment import RacingEnvironment
from settings import *

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(f'Autonomous Car - Deep Q Learning')

RaceEnv = RacingEnvironment()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    observation = RaceEnv.car.perceive()
    print(observation)
    RaceEnv.draw(screen)
    pygame.display.flip()