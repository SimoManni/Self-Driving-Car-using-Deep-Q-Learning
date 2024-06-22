import pygame
import sys
import numpy as np

from settings import WIDTH, HEIGHT, car_start_pos
from Car import Car
from Track import Track

# Initialize Pygame
pygame.init()

# Screen dimensions
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simple Car Game")

track = Track()
car = Car(screen, car_start_pos, track.contour_points)

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Get key states
    keys = pygame.key.get_pressed()

    # Update car
    car.update(keys)

    # Clear screen
    screen.fill((0, 0, 0))

    # Draw track
    track.draw(screen)
    track.draw_lines(screen)
    car.perceive()

    # Draw car
    car.draw(screen)
    car.check_collision()

    # Update display
    pygame.display.flip()

    # Cap the frame rate
    pygame.time.Clock().tick(30)

# Quit Pygame
pygame.quit()
sys.exit()