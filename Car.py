import pygame
import numpy as np

class Car(pygame.sprite.Sprite):
    def __init__(self, car_start_pos, contour_lines):
        super().__init__()

        # Load image and resize
        car_image = pygame.image.load('car.png')
        self.image = pygame.transform.scale(car_image, (50, 50))
        self.original_image = self.image
        self.rect = self.image.get_rect()

        # Initial position and velocity
        self.rect.center = car_start_pos
        self.speed = 0
        self.angle = 35

        # Parameters
        self.acceleration = 0.2
        self.brake_deceleration = 0.3
        self.friction = 0.1
        self.max_speed = 5

        # Line segments of the track
        self.contour_lines = np.vstack(contour_lines)

    def update(self, keys):
        # Handle rotation
        if keys[pygame.K_LEFT] and self.speed != 0:
            self.angle += 5
        if keys[pygame.K_RIGHT] and self.speed != 0:
            self.angle -= 5

        # Handle acceleration and braking
        if keys[pygame.K_UP]:
            self.speed += self.acceleration
            if self.speed > self.max_speed:
                self.speed = self.max_speed
        if keys[pygame.K_DOWN]:
            self.speed -= self.brake_deceleration
            if self.speed < -self.max_speed / 2:  # Allow some reverse speed
                self.speed = -self.max_speed / 2

        # Apply friction to gradually slow down the car
        if not keys[pygame.K_UP] and not keys[pygame.K_DOWN]:
            if self.speed > 0:
                self.speed -= self.friction
                if self.speed < 0:
                    self.speed = 0
            elif self.speed < 0:
                self.speed += self.friction
                if self.speed > 0:
                    self.speed = 0

        # Update position based on speed and angle
        self.rect.x += self.speed * pygame.math.Vector2(0, -1).rotate(-self.angle).x
        self.rect.y += self.speed * pygame.math.Vector2(0, -1).rotate(-self.angle).y

    def draw(self, surface):
        rotated_image = pygame.transform.rotate(self.original_image, self.angle)
        rect = rotated_image.get_rect(center=self.rect.center)
        surface.blit(rotated_image, rect.topleft)

    def reset(self, car_start_pos):
        self.rect.center = car_start_pos
        self.speed = 0
        self.angle = 0

    # def check_collision(self):
    #     top_left = self.rect.topleft
    #     top_right = self.rect.topright
    #     bottom_left = self.rect.bottomleft
    #     bottom_right = self.rect.bottomright
    #
    #     # Top edge
    #     m_top = (top_right[1] - top_left[1]) / (top_right[0] - top_left[0])
    #     b_top = top_left[1] - m_top * top_left[0]
    #
    #     # Right edge
    #     if bottom_right[0] - top_right[0] != 0:
    #         m_right = (bottom_right[1] - top_right[1]) / (bottom_right[0] - top_right[0])
    #         b_right = top_right[1] - m_right * top_right[0]
    #     else:
    #         m_right = 1e10
    #         b_right = -1e10
    #
    #     # Bottom edge
    #     m_bottom = (bottom_right[1] - bottom_left[1]) / (bottom_right[0] - bottom_left[0])
    #     b_bottom = bottom_left[1] - m_bottom * bottom_left[0]
    #
    #     # Left edge
    #     if bottom_left[0] - top_left[0] != 0:
    #         m_left = (bottom_left[1] - top_left[1]) / (bottom_left[0] - top_left[0])
    #         b_left = top_left[1] - m_left * top_left[0]
    #     else:
    #         m_left = -1e10
    #         b_left = 1e10
    #
    #     distances = self._lines_intersection([m_bottom, b_bottom])
    #     if np.min(distances) <= self.rect.width / 2:
    #         print('Collision')
    # def _lines_intersection(self, line_coeff):
    #     contour_line_coeff = self.contour_lines
    #     m_diff = line_coeff[0] - contour_line_coeff[:, 0]
    #     nonzero_indices = np.nonzero(m_diff)[0]
    #     x_inter = (contour_line_coeff[nonzero_indices, 1] - line_coeff[1]) / m_diff[nonzero_indices]
    #     y_inter = line_coeff[0] * x_inter + line_coeff[1]
    #
    #     points = np.column_stack((x_inter, y_inter)).astype(np.int32)
    #     for point in points:
    #         pygame.draw.circle(screen, (255, 0, 0), point, 5)
    #     distances_inter = np.sqrt(np.sum(np.square(points - self.rect.center), axis=1))
    #     distances_parallel = np.abs(contour_line_coeff[~nonzero_indices, 1] - line_coeff[1])
    #
    #     return np.concatenate((distances_inter, distances_parallel))