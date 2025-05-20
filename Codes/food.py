# File: food.py

import pygame
import random
# Import settings from config.py
from config import BLOCK_SIZE, SCREEN_WIDTH, SCREEN_HEIGHT, BLUE

class Food:
    def __init__(self):
        self.position = (0, 0)

    def spawn(self, occupied_positions):
        while True:
            x = random.randrange(0, SCREEN_WIDTH // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randrange(0, SCREEN_HEIGHT // BLOCK_SIZE) * BLOCK_SIZE
            self.position = (x, y)
            # Ensures food does not spawn INSIDE any snake
            if self.position not in occupied_positions:
                break

    def draw(self, screen):
        pygame.draw.rect(screen, BLUE, [self.position[0], self.position[1], BLOCK_SIZE, BLOCK_SIZE])

    def get_pos(self):
        return self.position
