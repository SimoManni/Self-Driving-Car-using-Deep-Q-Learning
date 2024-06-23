import pygame
import numpy as np
import cv2

from Car import Car
from settings import WIDTH, HEIGHT, car_start_pos

class RacingEnvironment:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Autonomous Car - Deep Q Learning")

        track = pygame.image.load('track.png')
        self.image = pygame.transform.scale(track, (WIDTH, HEIGHT))
        self.contour_points = self.barriers()
        self.checkpoints = self.define_checkpoints()

        self.car = Car(self.screen, car_start_pos, self.contour_points, self.checkpoints)

        self.VIS_BARRIERS = True
        self.VIS_CHECKPOINTS = True

    def update(self):
        keys = pygame.key.get_pressed()
        self.car.update(keys)
        self.screen.fill((0, 0, 0))

        self.draw()

        self.car.perceive()
        self.car.draw()
        self.car.check_collision()
        self.car.checkpoint()

        pygame.display.flip()
        pygame.time.Clock().tick(30)

    def draw(self):
        self.screen.blit(self.image, (0, 0))
        if self.VIS_BARRIERS:
            self.draw_lines()
        if self.VIS_CHECKPOINTS:
            self.draw_checkpoints()

    def draw_lines(self):
        pygame.draw.lines(self.screen, (255, 0, 0), True, self.contour_points[0], 5)
        pygame.draw.lines(self.screen, (255, 0, 0), True, self.contour_points[1], 5)
        for point in np.vstack(self.contour_points):
            pygame.draw.circle(self.screen, (0, 0, 255), point, 5)
    def draw_checkpoints(self):
        for line in self.checkpoints:
            pygame.draw.line(self.screen, (255, 0, 0), line[:2], line[2:], 5)

    def define_checkpoints(self):
        return np.array([[184, 550, 200, 470],
                             [165, 461, 102, 507],
                             [160, 441, 105, 385],
                             [207, 450, 220, 371],
                             [237, 365, 315, 377],
                             [197, 325, 250, 271],
                             [148, 203, 224, 220],
                             [286, 208, 329, 141],
                             [379, 301, 428, 241],
                             [482, 265, 546, 310],
                             [451, 166, 527, 163],
                             [555, 129, 566, 51],
                             [605, 172, 676, 148],
                             [622, 288, 697, 284],
                             [628, 411, 703, 414],
                             [595, 464, 633, 532],
                             [485, 472, 489, 547],
                             [314, 480, 312, 558]])

    def barriers(self):
        image = cv2.imread('track.png')
        image = cv2.resize(image, (800, 600))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Extract contours with minimum length to prevent other lines except for the contours of the track to be selected
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        min_contour_length = 200
        long_contours = [contour for contour in contours if cv2.arcLength(contour, True) > min_contour_length]

        # Outer and inner contour of the track
        epsilon_outer = 0.001 * cv2.arcLength(long_contours[0], True)
        approx_contours_outer = cv2.approxPolyDP(long_contours[0], epsilon_outer, True)
        approx_contours_outer = np.squeeze(approx_contours_outer)

        epsilon_inner = 0.001 * cv2.arcLength(long_contours[-1], True)
        approx_contours_inner = cv2.approxPolyDP(long_contours[-1], epsilon_outer, True)
        approx_contours_inner = np.squeeze(approx_contours_inner)

        return [approx_contours_outer, approx_contours_inner]