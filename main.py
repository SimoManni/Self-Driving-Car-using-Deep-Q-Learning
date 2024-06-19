import pygame
import sys

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simple Car Game")

# Load images
track = pygame.image.load('track.png')
track = pygame.transform.scale(track, (800, 600))
car_image = pygame.image.load('car.png')
car_image = pygame.transform.scale(car_image, (50, 50))

class Car:
    def __init__(self, image, x, y):
        self.image = image
        self.x = x
        self.y = y
        self.speed = 0
        self.angle = 0
        self.acceleration = 0.2
        self.brake_deceleration = 0.3
        self.friction = 0.1
        self.max_speed = 5

    def update(self, keys):
        # Handle rotation
        if keys[pygame.K_LEFT]:
            self.angle += 5
        if keys[pygame.K_RIGHT]:
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
        self.x += self.speed * pygame.math.Vector2(0, -1).rotate(-self.angle).x
        self.y += self.speed * pygame.math.Vector2(0, -1).rotate(-self.angle).y

    def draw(self, surface):
        rotated_image = pygame.transform.rotate(self.image, self.angle)
        rect = rotated_image.get_rect(center=(self.x, self.y))
        surface.blit(rotated_image, rect.topleft)

# Create a car instance
car = Car(car_image, WIDTH // 7, 4 * HEIGHT // 5)

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
    screen.blit(track, (0, 0))

    # Draw car
    car.draw(screen)

    # Update display
    pygame.display.flip()

    # Cap the frame rate
    pygame.time.Clock().tick(30)

# Quit Pygame
pygame.quit()
sys.exit()