import pygame

import numpy as np


class KeyboardAgent:
    def __init__(self):
        pygame.init()

        self._location1 = 0
        self._location2 = 0

    def fit(self, dataset):
        return

    def draw_action(self, state):
        events = pygame.event.get()

        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self._location1 += 1
                if event.key == pygame.K_RIGHT:
                    self._location1 -= 1
                if event.key == pygame.K_UP:
                    self._location2 += 1

                if event.key == pygame.K_SPACE:

                    cont = True
                    while cont:
                        new_events = pygame.event.get()
                        for new_event in new_events:
                            if new_event.type == pygame.KEYUP:
                                if new_event.key == pygame.K_SPACE:
                                    cont = False





            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    self._location1 -= 1
                if event.key == pygame.K_RIGHT:
                    self._location1 += 1
                if event.key == pygame.K_UP:
                    self._location2 -= 1

        omega = 0.16*self._location1
        v = 0.1*self._location2
        return np.array([v, omega])

    def episode_start(self):

        pass

    def stop(self):
        pass