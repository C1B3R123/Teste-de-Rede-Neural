import pygame
from collections import deque
import numpy as np
from config import TAMANHO_BLOCO, LARGURA_TELA, ALTURA_TELA, VERDE, CIMA, BAIXO, ESQUERDA, DIREITA, INPUT_NEURONS

class Snake:
    def __init__(self):
        self.body = deque([(LARGURA_TELA // 2, ALTURA_TELA // 2)]) 
        self.direction = DIREITA 
        self.grow = False
        self.score = 0
        self.lifespan = 0
        self.is_alive = True

    def change_direction(self, new_direction):
        # Impede virar 180 graus instantaneamente
        if (new_direction[0] * -1, new_direction[1] * -1) != self.direction:
            self.direction = new_direction

    def move(self):
        if not self.is_alive:
            return

        head_x, head_y = self.body[0]
        dir_x, dir_y = self.direction
        new_head = (head_x + dir_x * TAMANHO_BLOCO, head_y + dir_y * TAMANHO_BLOCO)
        self.body.appendleft(new_head) 

        if not self.grow:
            self.body.pop() 
        else:
            self.grow = False

        self.lifespan += 1
    def ate_comida(self):
        self.grow = True
        self.score += 1

    def check_collision(self):
        if not self.is_alive:
            return False 
            
        head = self.body[0]
        if head[0] >= LARGURA_TELA or head[0] < 0 or head[1] >= ALTURA_TELA or head[1] < 0:
            self.is_alive = False
            return True
        if len(self.body) > 1 and head in list(self.body)[1:]:
             self.is_alive = False
             return True
        return False

    def draw(self, screen):
        if not self.is_alive:
            return
        for segment in self.body:
            pygame.draw.rect(screen, VERDE, [segment[0], segment[1], TAMANHO_BLOCO, TAMANHO_BLOCO])

    def get_head_pos(self):
        return self.body[0]

    def get_body_pos(self):
        return list(self.body)

    # Métodos para a IA 
    def get_state_for_nn(self, food_pos):
        # Esta função coleta as informações do jogo para alimentar a rede neural
        head_x, head_y = self.get_head_pos()
        food_x, food_y = food_pos
        current_direction = self.direction

        # Define as direções relativas em relação à direção atual da cobra
        if current_direction == DIREITA:
            ahead = DIREITA
            left = CIMA
            right = BAIXO
        elif current_direction == ESQUERDA:
            ahead = ESQUERDA
            left = BAIXO
            right = CIMA
        elif current_direction == CIMA:
            ahead = CIMA
            left = ESQUERDA
            right = DIREITA
        else:
            ahead = BAIXO
            left = DIREITA
            right = ESQUERDA

        #  Sensores de Perigo (Parede ou Corpo) 
        
        danger_ahead = self._is_danger(head_x + ahead[0] * TAMANHO_BLOCO, head_y + ahead[1] * TAMANHO_BLOCO)
        danger_left = self._is_danger(head_x + left[0] * TAMANHO_BLOCO, head_y + left[1] * TAMANHO_BLOCO)
        danger_right = self._is_danger(head_x + right[0] * TAMANHO_BLOCO, head_y + right[1] * TAMANHO_BLOCO)
        # Direção da comida em relação à cabeça da cobra 
        food_up = int(food_y < head_y)
        food_down = int(food_y > head_y)
        food_left = int(food_x < head_x)
        food_right = int(food_x > head_x)

        # Direção Atual 
        dir_left = int(current_direction == ESQUERDA)
        dir_right = int(current_direction == DIREITA)
        dir_up = int(current_direction == CIMA)
        dir_down = int(current_direction == BAIXO)

        # Monta o vetor de entradas para a rede neural
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
        ]).reshape(1, INPUT_NEURONS) # Reshape para o formato (1, input_size) que a NN espera

        return inputs

    def _is_danger(self, x, y):
        # Verifica se uma posição (x, y) representa perigo (parede ou corpo)
        if x >= LARGURA_TELA or x < 0 or y >= ALTURA_TELA or y < 0:
            return 1 
        if (x, y) in list(self.body):
            return 1
        return 0 

    def get_fitness(self):
        return self.score * 100 + self.lifespan
