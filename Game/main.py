import pygame
import sys
import numpy as np

from settings import WIDTH, HEIGHT, car_start_pos
from Car import Car
from RacingEnvironment import RacingEnvironment

# Initialize Pygame
pygame.init()

# Screen dimensions
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simple Car Game")

RaceEnv = RacingEnvironment()

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    RaceEnv.update()

# Quit Pygame
pygame.quit()
sys.exit()