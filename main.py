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

        observation_, reward, done = game.step(0)
        observation = np.array(observation_)

        gtime = 0  # set game time back to 0

        renderFlag = False  # if you want to render every episode set to true

        if e % 10 == 0 and e > 0:  # render every 10 episodes
            renderFlag = True

        while not done:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return


            action = agent.choose_action(observation)
            observation_, reward, done = game.step(action)
            observation_ = np.array(observation_)

            # This is a countdown if no reward is collected the car will be done within 100 ticks
            if reward == 0:
                counter += 1
                if counter > 100:
                    done = True
            else:
                counter = 0

            score += reward

            agent.remember(observation, action, reward, observation_, int(done))
            observation = observation_
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

        print('episode: ', e, 'score: %.2f' % score,
              ' average score %.2f' % avg_score,
              ' epsilon: ', agent.epsilon,
              ' memory size', agent.memory.mem_cntr % MEM_SIZE)


run()