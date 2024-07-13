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

# Main game loop
start_time = pygame.time.get_ticks()
running = False
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
        elif event.type == pygame.KEYDOWN and not running:
            # Start the game when any key is pressed for the first time
            running = True

    if running:
        # Game logic when running
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
    else:
        # Display initial screen with instructions
        RaceEnv.draw()
        RaceEnv.write(score)
        pygame.display.update()

        text = FONT.render('Press any key to start', True, (0, 0, 0))
        text_rect = text.get_rect(center=(460, 410))
        screen.blit(text, text_rect)
        pygame.display.flip()  # Update the full display

pygame.quit()