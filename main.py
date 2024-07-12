import pygame
import numpy as np

from settings import *
from Environment import SimulationEnvironment
from Q_learning import Agent


game = SimulationEnvironment()

MAX_TIME_SECONDS = 5
GameTime = 0
GameHistory = []

agent = Agent()
# agent.load_model()

ddqn_scores = []
eps_history = []


def run():
    for e in range(N_EPISODES):

        # if e % 100 == 0:
        #     simulate_agent(e)

        game.reset()
        score = 0
        counter = np.zeros(N_CARS)
        gtime = 0

        state_prev, reward, done = game.step(np.zeros(N_CARS))

        while not np.any(done):
            actions = agent.get_actions(state_prev)
            new_state, reward, done = game.step(actions)

            # Countdown if no reward is collected
            counter[reward == 0] += 1
            if np.any(counter > 100):
                done[counter > 100] = True

            score += np.mean(reward)

            agent.remember(state_prev, actions, reward, new_state, done.astype(int))
            agent.learn()
            state_prev = new_state

            gtime += 1

            if gtime >= TOTAL_GAMETIME:
                done = True

        if e % 10 == 0 and e > 10:
            agent.save_model()
            print("save model")


        eps_history.append(agent.epsilon)
        ddqn_scores.append(score)
        avg_score = np.mean(ddqn_scores[max(0, e - 100):(e + 1)])

        print('Episode: ', e,
              ', Score: %.2f' % score,
              ', Average score %.2f' % avg_score,
              ', Epsilon: ', agent.epsilon,
              ', Memory size', agent.memory.mem_cntr % MEM_SIZE)


run()