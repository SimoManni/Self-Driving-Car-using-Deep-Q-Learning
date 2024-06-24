import pygame
import numpy as np

from settings import *
from RacingEnvironment import RacingEnvironment
from Q_learning import Agent


game = RacingEnvironment()

GameTime = 0
GameHistory = []
renderFlag = False

agent = Agent()

ddqn_scores = []
eps_history = []


def run():
    for e in range(N_EPISODES):

        game.reset()  # reset env

        done = False
        score = 0
        counter = 0

        state_prev, reward, done = game.step(0)
        game.draw()

        gtime = 0  # set game time back to 0

        renderFlag = False

        if e % 10 == 0 and e > 0:
            renderFlag = True

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return


            action = agent.choose_action(state_prev)
            observation, reward, done = game.step(action)

            # This is a countdown if no reward is collected the car will be done within 100 ticks
            if reward == 0:
                counter += 1
                if counter > 100:
                    done = True
            else:
                counter = 0

            score += reward

            agent.remember(state_prev, action, reward, observation, int(done))
            agent.learn()

            gtime += 1

            if gtime >= TOTAL_GAMETIME:
                done = True

            if renderFlag:
                game.draw()

        eps_history.append(agent.epsilon)
        ddqn_scores.append(score)
        avg_score = np.mean(ddqn_scores[max(0, e - 100):(e + 1)])

        if e % REPLACE_TARGET == 0 and e > REPLACE_TARGET:
            agent.update_network_parameters()

        if e % 10 == 0 and e > 10:
            agent.save_model()
            print("save model")

        print('Episode: ', e,
              ', Score: %.2f' % score,
              ', Average score %.2f' % avg_score,
              ', Epsilon: ', agent.epsilon,
              ', Memory size', agent.memory.mem_cntr % MEM_SIZE)


run()