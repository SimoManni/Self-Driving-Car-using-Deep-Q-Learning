import pygame

from settings import *
from AutonomousCar import AutonomousCar
from Q_learning import Agent

class SimulationEnvironment:
    def __init__(self):
        track = pygame.image.load('track.png')
        self.image = pygame.transform.scale(track, (WIDTH, HEIGHT))
        self.contour_points = BARRIERS

        self.cars = []
        for i in range(N_CARS):
            car = AutonomousCar()
            self.cars.append(car)

        self.VIS_BARRIERS = True
        self.VIS_CHECKPOINTS = True

    def step(self, actions):
        new_state_list = []
        reward_list = []
        done_list = []

        for car, action in zip(self.cars, actions):
            done = False
            car.update(action)
            reward = LIFE_REWARD

            # Check if car passes checkpoint
            if car.checkpoint():
                reward += GOAL_REWARD

            # Check if collision occurred
            if car.check_collision():
                reward += PENALTY
                done = True

            new_state = car.get_state()
            if done:
                new_state = np.full(INPUT_DIMS, np.nan)

            new_state_list.append(new_state)
            reward_list.append(reward)
            done_list.append(done)

        return np.array(new_state_list), np.array(reward_list), np.array(done_list)

    def get_state(self):
        new_states = []
        for car in self.cars:
            new_states.append(car.get_state())
        return np.array(new_states)

    def reset(self):
        for car in self.cars:
            car.reset()

    # Functions for visualization
    def draw(self, screen):
        screen.blit(self.image, (0, 0))
        screen.blit(self.image, (0, 0))
        if self.VIS_BARRIERS:
            self.draw_lines(screen)
        if self.VIS_CHECKPOINTS:
            self.draw_checkpoints(screen)

        for car in self.cars:
            car.draw(screen)

    def draw_lines(self, screen):
        pygame.draw.lines(screen, (255, 0, 0), True, self.contour_points[0], 5)
        pygame.draw.lines(screen, (255, 0, 0), True, self.contour_points[1], 5)
        for point in np.vstack(self.contour_points):
            pygame.draw.circle(screen, (0, 0, 255), point, 5)

    def draw_checkpoints(self, screen):
        for line in CHECKPOINTS:
                pygame.draw.line(screen, (255, 0, 0), line[:2], line[2:], 3)

class RacingEnvironment:
    def __init__(self, screen):
        self.screen = screen
        track = pygame.image.load('track.png')
        self.image = pygame.transform.scale(track, (WIDTH, HEIGHT))
        self.contour_points = BARRIERS

        self.car = AutonomousCar(random=False)
        self.agent = Agent()
        self.agent.load_model()

        self.VIS_BARRIERS = True
        self.VIS_CHECKPOINTS = True
        self.FONT = pygame.font.Font(None, 32)

    def get_state(self):
        return self.car.get_state()

    def get_action(self, state):
        q_values = self.agent.policy_dqn.predict(state)
        return np.argmax(q_values)

    def update(self, action):

        done = False
        self.car.update(action)
        reward = LIFE_REWARD

        # Check if car passes checkpoint
        if self.car.checkpoint():
            reward += GOAL_REWARD

        # Check if collision occurred
        if self.car.check_collision():
            reward += PENALTY
            done = True

        return reward, done

    def reset(self):
        self.car.reset()

    def write(self, score):
        txt_score = self.FONT.render(f'Score: {score}', True, (0, 0, 0))
        self.screen.blit(txt_score, (50, 50))

        txt_speed = self.FONT.render(f'Speed: {self.car.speed:.1f}', True, (0, 0, 0))
        self.screen.blit(txt_speed, (50, 70))

# Functions for visualization
    def draw(self):
        self.screen.blit(self.image, (0, 0))
        self.screen.blit(self.image, (0, 0))
        text = pygame.font.Font(None, 30).render(f"Laps completed: {self.car.laps}", True, (0, 0, 0))
        if self.VIS_BARRIERS:
            self.draw_lines()
        if self.VIS_CHECKPOINTS:
            self.draw_checkpoints()

        self.car.draw(self.screen)

    def draw_lines(self):
        pygame.draw.lines(self.screen, (255, 0, 0), True, self.contour_points[0], 5)
        pygame.draw.lines(self.screen, (255, 0, 0), True, self.contour_points[1], 5)
        for point in np.vstack(self.contour_points):
            pygame.draw.circle(self.screen, (0, 0, 255), point, 5)
    def draw_checkpoints(self):
        for idx, line in enumerate(self.car.checkpoints):
            if idx < self.car.passed_checkpoints:
                pygame.draw.line(self.screen, (255, 255, 0), line[:2], line[2:], 3)
            else:
                pygame.draw.line(self.screen, (255, 0, 0), line[:2], line[2:], 3)


