import pygame
import numpy as np
import cv2

from settings import WIDTH, HEIGHT

class Track:
    def __init__(self):
        track = pygame.image.load('track.png')
        self.image = pygame.transform.scale(track, (WIDTH, HEIGHT))
        self.contour_points = self.barriers()
        self.vis = True

    def draw(self, screen):
        screen.blit(self.image, (0, 0))

    def draw_lines(self, screen):
        if self.vis:
            pygame.draw.lines(screen, (255, 0, 0), True, self.contour_points[0], 5)
            pygame.draw.lines(screen, (255, 0, 0), True, self.contour_points[1], 5)

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