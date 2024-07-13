import pygame
import numpy as np

from settings import *

class AutonomousCar():
    def __init__(self, random=True):
        # Load image and resize
        car_image = pygame.image.load('car.png')
        self.image = pygame.transform.scale(car_image, (20, 40))
        self.original_image = self.image
        self.rect = self.image.get_rect()

        # Initial position and velocity
        if random:
            self.car_start_pos = None
            self.angle = None
            self.checkpoints = self.get_checkpoints_and_start_pos()
        else:
            self.car_start_pos = INIT_POS
            self.angle = INIT_ANGLE
            self.checkpoints = CHECKPOINTS
        self.rect.center = self.car_start_pos
        self.speed = 0
        self.car_start_pos_original = self.car_start_pos
        self.angle_original = self.angle

        self.corners = np.array([
                [-self.rect.width / 2, -self.rect.height / 2],  # Top-left
                [self.rect.width / 2, -self.rect.height / 2],   # Top-right
                [self.rect.width / 2, self.rect.height / 2],    # Bottom-right
                [-self.rect.width / 2, self.rect.height / 2]    # Bottom-left
            ]) + np.array(self.rect.center)

        # Parameters
        self.acceleration = 0.2
        self.brake_deceleration = 0.3
        self.friction = 0.1
        self.max_speed = 5

        # Line segments of the track
        self.contour_points = BARRIERS
        self.contour_lines = self.get_line_segments()
        self.passed_checkpoints = 0
        self.laps = 0
        self.perceived_points = None

        self.VIS_PERCEPTION = True


    def update(self, action):
        if action == 0:
            # Accelerate forward
            self.speed += self.acceleration
            if self.speed > self.max_speed:
                self.speed = self.max_speed

        elif action == 1:
            # Accelerate forward and rotate right
            self.speed += self.acceleration
            if self.speed > self.max_speed:
                self.speed = self.max_speed
            if self.speed != 0:
                self.angle -= 5
        elif action == 2:
            # Accelerate forward and rotate left
            self.speed += self.acceleration
            if self.speed > self.max_speed:
                self.speed = self.max_speed
            if self.speed != 0:
                self.angle += 5
        elif action == 3:
            # Turn right
            if self.speed != 0:
                self.angle -= 5
        elif action == 4:
            # Turn left
            if self.speed != 0:
                self.angle += 5

        # Apply friction to gradually slow down the car if gas is not applied
        if action == 3 or action == 4 or action == 5:
            self.speed -= self.friction
            if self.speed < 0:
                self.speed = 0

        # Update position based on speed and angle
        self.rect.x += self.speed * pygame.math.Vector2(0, -1).rotate(-self.angle).x
        self.rect.y += self.speed * pygame.math.Vector2(0, -1).rotate(-self.angle).y
        self.rotate()

    def rotate(self):
        angle = self.angle * np.pi / 180

        # Calculate rotation matrix
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])

        vertices_rel = np.array([
                        [-self.rect.width / 2, -self.rect.height / 2],  # Top-left
                        [self.rect.width / 2, -self.rect.height / 2],   # Top-right
                        [self.rect.width / 2, self.rect.height / 2],    # Bottom-right
                        [-self.rect.width / 2, self.rect.height / 2]    # Bottom-left
                    ])
        rotated_vertices_rel = np.dot(vertices_rel, rotation_matrix)
        rotated_vertices = rotated_vertices_rel + np.array(self.rect.center)

        self.corners = rotated_vertices

    def reset(self):
        self.rect.center = self.car_start_pos_original
        self.angle = self.angle_original
        self.speed = 0
        self.passed_checkpoints = 0
        self.laps = 0

    def check_collision(self):
        corners = np.array(self.corners, dtype='int32')
        top_left = corners[0]
        top_right = corners[1]
        bottom_right = corners[2]
        bottom_left = corners[3]

        lines = np.array([
            np.concatenate([top_left, top_right]),  # Line from top_left to top_right
            np.concatenate([top_right, bottom_right]),  # Line from top_right to bottom_right
            np.concatenate([bottom_right, bottom_left]),  # Line from bottom_right to bottom_left
            np.concatenate([bottom_left, top_left])  # Line from bottom_left to top_left
        ])

        # Calculate denominator for all pairs of lines
        for line in lines:
            den = ((self.contour_lines[:, 0] - self.contour_lines[:, 2]) *
                   (line[1] - line[3]) -
                   (self.contour_lines[:, 1] - self.contour_lines[:, 3]) *
                   (line[0] - line[2]))

            # Find indices where den is not zero (to avoid division by zero)
            non_zero_indices = np.nonzero(den)

            # Calculate t and u for all pairs of lines where den is not zero
            t_numerators = ((self.contour_lines[non_zero_indices, 0] - line[0]) *
                            (line[1] - line[3]) -
                            (self.contour_lines[non_zero_indices, 1] - line[1]) *
                            (line[0] - line[2]))

            t_denominators = den[non_zero_indices]

            u_numerators = -((self.contour_lines[non_zero_indices, 0] - self.contour_lines[non_zero_indices, 2]) *
                             (self.contour_lines[non_zero_indices, 1] - line[1]) -
                             (self.contour_lines[non_zero_indices, 1] - self.contour_lines[non_zero_indices, 3]) *
                             (self.contour_lines[non_zero_indices, 0] - line[0]))

            u_denominators = den[non_zero_indices]

            t = t_numerators / t_denominators
            u = u_numerators / u_denominators

            collision_mask = (t > 0) & (t < 1) & (u > 0) & (u < 1)

            if np.any(collision_mask):
                return True
            return False

    def get_state(self):
        distances = self.perceive()
        return np.concatenate((distances, [self.speed]))

    def perceive(self):

        corners = self.corners
        midpoints = (corners + np.roll(corners, -1, axis=0)) / 2
        points = np.vstack((corners, midpoints))

        center = self.rect.center
        extension_factor = 7
        vectors = points - center
        extended_vectors = vectors * extension_factor
        extended_points = center + extended_vectors

        perceived_points = []
        for point in extended_points:
            inter_point = self.get_line_intersection(np.concatenate((center, point)))
            if inter_point is not None:
                perceived_points.append(inter_point)
            else:
                perceived_points.append(point)

        perceived_points = np.array(perceived_points)
        self.perceived_points = perceived_points
        distances = np.sqrt(np.sum(np.square(perceived_points - center), axis=1))

        return distances

    def checkpoint(self):
        corners = np.array(self.corners, dtype='int32')

        top_left = corners[0]
        top_right = corners[1]
        bottom_right = corners[2]
        bottom_left = corners[3]

        lines = np.array([
            np.concatenate([top_right, bottom_right]),  # Line from top_right to bottom_right
            np.concatenate([bottom_left, top_left])  # Line from bottom_left to top_left
        ])

        if self.passed_checkpoints == 0:
            checkpoint = self.checkpoints[0]
        elif self.passed_checkpoints == self.checkpoints.shape[0]:
            checkpoint = self.checkpoints[-1]
            self.passed_checkpoints = 0
        else:
            checkpoint = self.checkpoints[self.passed_checkpoints]

        den = ((checkpoint[0] - checkpoint[2]) *
               (lines[:, 1] - lines[:, 3]) -
               (checkpoint[1] - checkpoint[3]) *
               (lines[:, 0] - lines[:, 2]))

        non_zero_indices = np.nonzero(den)
        den = den[non_zero_indices]

        # Calculate t and u
        t_numerators = ((checkpoint[0] - lines[non_zero_indices, 0]) *
                        (lines[non_zero_indices, 1] - lines[non_zero_indices, 3]) -
                        (checkpoint[1] - lines[non_zero_indices, 1]) *
                        (lines[non_zero_indices, 0] - lines[non_zero_indices, 2]))


        u_numerators = -((checkpoint[0] - checkpoint[2]) *
                         (checkpoint[1] - lines[non_zero_indices, 1]) -
                         (checkpoint[1] - checkpoint[3]) *
                         (checkpoint[0] - lines[non_zero_indices, 0]))

        t = t_numerators / den
        u = u_numerators / den

        collision = (t > 0) & (t < 1) & (u > 0) & (u < 1)
        if np.any(collision):
            self.passed_checkpoints += 1
            if self.passed_checkpoints == self.checkpoints.shape[0]:
                self.passed_checkpoints = 0
                self.laps += 1
            return True

        return False

    def get_line_intersection(self, line):
        x3, y3, x4, y4 = line
        x1 = self.contour_lines[:, 0]
        y1 = self.contour_lines[:, 1]
        x2 = self.contour_lines[:, 2]
        y2 = self.contour_lines[:, 3]

        # Find denominator
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        valid = den != 0

        # Exclude parallel lines
        x1 = x1[valid]
        y1 = y1[valid]
        x2 = x2[valid]
        y2 = y2[valid]
        den = den[valid]

        # Compute intersection points
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den
        valid_t = (t >= 0) & (t <= 1)
        valid_u = (u >= 0) & (u <= 1)
        valid_intersections = valid_t & valid_u

        x1 = x1[valid_intersections]
        y1 = y1[valid_intersections]
        x2 = x2[valid_intersections]
        y2 = y2[valid_intersections]
        t = t[valid_intersections]

        if len(t) == 0:
            return None
        else:
            pts = np.vstack((x1 + t * (x2 - x1), y1 + t * (y2 - y1))).T
            pts = np.floor(pts).astype(int)
            if len(pts) == 1:
                return np.array(pts[0])
            else:
                distances = np.sqrt(np.sum(np.square(pts - self.rect.center), axis=1))
                idx = np.argmin(distances)
                return np.array(pts[idx])

    def get_line_segments(self):
        # Function definition for creating lines
        def extract_line_segments(contours):
            line_segments = []
            for i in range(len(contours) - 1):
                line_segments.append(np.concatenate((contours[i], contours[i + 1])))
            line_segments.append(np.concatenate((contours[-1], contours[0])))
            return np.array(line_segments)


        line_segments_outer = extract_line_segments(self.contour_points[0])
        line_segments_inner = extract_line_segments(self.contour_points[1])

        return np.vstack((line_segments_outer, line_segments_inner))

    def draw(self, screen):
        if self.perceived_points is not None and self.VIS_PERCEPTION:
            for point in self.perceived_points:
                pygame.draw.line(screen, (0, 255, 0), self.rect.center, point, 3)
                pygame.draw.circle(screen, (0, 0, 255), point, 3)

        rotated_image = pygame.transform.rotate(self.original_image, self.angle)
        rect = rotated_image.get_rect(center=self.rect.center)
        screen.blit(rotated_image, rect.topleft)

    def get_checkpoints_and_start_pos(self):
        checkpoints = CHECKPOINTS
        index = np.random.randint(0, len(checkpoints))

        # Definition of starting position and angle
        self.car_start_pos = STARTING_POINTS[index]
        self.rect.center = self.car_start_pos
        self.angle = STARTING_ANGLES[index]

        # Definition of checkpoints
        if index != 0:
            return np.vstack((checkpoints[index:], checkpoints[:index]))
        elif index == 0:
            return checkpoints
