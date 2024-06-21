import pygame
import sys
import numpy as np

from settings import WIDTH, HEIGHT
from Car import Car
from Track import Track

# Initialize Pygame
pygame.init()

# Screen dimensions
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simple Car Game")

contour_points_outer = np.loadtxt('contour_points_outer.txt')
contour_points_inner = np.loadtxt('contour_points_inner.txt')
car_start_pos = (WIDTH // 7, 4 * HEIGHT // 5)

car = Car(car_start_pos, [contour_points_outer, contour_points_inner])
track = Track()

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

    # Draw car
    car.draw(screen)

    # Update display
    pygame.display.flip()

    # Cap the frame rate
    pygame.time.Clock().tick(30)

# Quit Pygame
pygame.quit()
sys.exit()