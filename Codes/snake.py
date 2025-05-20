import pygame
from collections import deque
import numpy as np
from config import BLOCK_SIZE, SCREEN_WIDTH, SCREEN_HEIGHT, GREEN, UP, DOWN, LEFT, RIGHT, INPUT_NEURONS

class Snake:
    def __init__(self, start_pos, start_direction, color):
        self.body = deque([start_pos])
        self.direction = start_direction
        self.color = color 
        self.grow = False
        self.score = 0
        self.lifespan = 0 
        self.is_alive = True

    def change_direction(self, new_direction):
        if self.is_alive and (new_direction[0] * -1, new_direction[1] * -1) != self.direction:
            self.direction = new_direction

    def move(self):
        if not self.is_alive:
            return

        head_x, head_y = self.body[0]
        dir_x, dir_y = self.direction
        new_head = (head_x + dir_x * BLOCK_SIZE, head_y + dir_y * BLOCK_SIZE)
        self.body.appendleft(new_head)

        if not self.grow:
            self.body.pop() 
        else:
            self.grow = False

        self.lifespan += 1 

    def ate_food(self):
        self.grow = True
        self.score += 1

    def check_collision(self, other_snake_body=None):
        if not self.is_alive:
            return False

        head = self.body[0]
        if head[0] >= SCREEN_WIDTH or head[0] < 0 or head[1] >= SCREEN_HEIGHT or head[1] < 0:
            self.is_alive = False
            return True
        if len(self.body) > 1 and head in list(self.body)[1:]:
             self.is_alive = False
             return True
        if other_snake_body and head in other_snake_body:
            self.is_alive = False
            return True
        return False

    def draw(self, screen):
        if not self.is_alive:
            return
        for segment in self.body:
            pygame.draw.rect(screen, self.color, [segment[0], segment[1], BLOCK_SIZE, BLOCK_SIZE])

    def get_head_pos(self):
        return self.body[0]

    def get_body_pos(self):
        return list(self.body)

    # --- Methods for AI ---
    def get_state_for_nn(self, food_pos, other_snake_body=None):
        head_x, head_y = self.get_head_pos()
        food_x, food_y = food_pos
        current_direction = self.direction
        
        if current_direction == RIGHT:
            ahead = RIGHT
            left = UP
            right = DOWN
        elif current_direction == LEFT:
            ahead = LEFT
            left = DOWN
            right = UP
        elif current_direction == UP:
            ahead = UP
            left = LEFT
            right = RIGHT
        else: 
            ahead = DOWN
            left = RIGHT
            right = LEFT

        danger_ahead = self._is_danger(head_x + ahead[0] * BLOCK_SIZE, head_y + ahead[1] * BLOCK_SIZE, other_snake_body)
        danger_left = self._is_danger(head_x + left[0] * BLOCK_SIZE, head_y + left[1] * BLOCK_SIZE, other_snake_body)
        danger_right = self._is_danger(head_x + right[0] * BLOCK_SIZE, head_y + right[1] * BLOCK_SIZE, other_snake_body)

        food_up = int(food_y < head_y)
        food_down = int(food_y > head_y)
        food_left = int(food_x < head_x)
        food_right = int(food_x > head_x)

        dir_left = int(current_direction == LEFT)
        dir_right = int(current_direction == RIGHT)
        dir_up = int(current_direction == UP)
        dir_down = int(current_direction == DOWN)

        inputs = np.array([
            danger_ahead,
            danger_left,
            danger_right,
            food_up,
            food_down,
            food_left,
            food_right,
            dir_left,
            dir_right,
            dir_up,
            dir_down
        ]).reshape(1, INPUT_NEURONS)
        return inputs

    def _is_danger(self, x, y, other_snake_body=None):
        if x >= SCREEN_WIDTH or x < 0 or y >= SCREEN_HEIGHT or y < 0:
            return 1 
        if (x, y) in list(self.body):
            return 1if other_snake_body and (x, y) in other_snake_body:
            return 1 
        return 0 

    def get_fitness(self):
        return self.score * 100 + self.lifespan
