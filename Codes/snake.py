import pygame
from collections import deque
import numpy as np
# Importa configurações do arquivo config.py
from config import TAMANHO_BLOCO, LARGURA_TELA, ALTURA_TELA, VERDE, CIMA, BAIXO, ESQUERDA, DIREITA, INPUT_NEURONS

class Snake:
    def __init__(self):
        self.body = deque([(LARGURA_TELA // 2, ALTURA_TELA // 2)]) # Começa no centro
        self.direction = DIREITA # Começa indo para a direita
        self.grow = False
        self.score = 0
        self.lifespan = 0 # Usado para o fitness na IA
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
        self.body.appendleft(new_head) # Adiciona a nova cabeça

        if not self.grow:
            self.body.pop() # Remove a cauda apenas se não for crescer
        else:
            self.grow = False

        self.lifespan += 1 # Aumenta o tempo de vida a cada movimento

    def ate_comida(self):
        self.grow = True
        self.score += 1

    def check_collision(self):
        if not self.is_alive:
            return False # Já está morta

        head = self.body[0]
        # Colisão com a parede
        if head[0] >= LARGURA_TELA or head[0] < 0 or head[1] >= ALTURA_TELA or head[1] < 0:
            self.is_alive = False
            return True
        # Colisão com o próprio corpo (começando do 2º segmento)
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

    # --- Métodos para a IA ---
    def get_state_for_nn(self, food_pos):
        # Esta função coleta as informações do jogo (sensores) para alimentar a rede neural
        head_x, head_y = self.get_head_pos()
        food_x, food_y = food_pos
        current_direction = self.direction

        # Define as direções relativas (em relação à direção atual da cobra)
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
        else: # current_direction == BAIXO
            ahead = BAIXO
            left = DIREITA
            right = ESQUERDA

        # --- Sensores de Perigo (Parede ou Corpo) ---
        # Verifica se há perigo em uma unidade de TAMANHO_BLOCO nas direções relativas
        danger_ahead = self._is_danger(head_x + ahead[0] * TAMANHO_BLOCO, head_y + ahead[1] * TAMANHO_BLOCO)
        danger_left = self._is_danger(head_x + left[0] * TAMANHO_BLOCO, head_y + left[1] * TAMANHO_BLOCO)
        danger_right = self._is_danger(head_x + right[0] * TAMANHO_BLOCO, head_y + right[1] * TAMANHO_BLOCO)

        # --- Localização da Comida ---
        # Direção da comida em relação à cabeça da cobra (binário)
        food_up = int(food_y < head_y)
        food_down = int(food_y > head_y)
        food_left = int(food_x < head_x)
        food_right = int(food_x > head_x)

        # --- Direção Atual ---
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
        # Colisão com a parede?
        if x >= LARGURA_TELA or x < 0 or y >= ALTURA_TELA or y < 0:
            return 1 # Sim, é perigo
        # Colisão com o corpo?
        # Verifica se a posição (x, y) está em algum segmento do corpo
        # Excluímos a própria cabeça se a posição for a da cabeça, para evitar detecção instantânea
        if (x, y) in list(self.body):
            return 1 # Sim, é perigo
        return 0 # Não, não é perigo

    def get_fitness(self):
        # Função de fitness para o algoritmo genético
        # Pontuação alta e tempo de vida longo são bons
        # Uma fórmula comum é score * C1 + lifespan * C2
        # C1 >> C2 para priorizar comida sobre simplesmente sobreviver sem comer
        # Também podemos adicionar uma penalidade por se aproximar da comida sem pegá-la,
        # ou um bônus por pegar comida rápido.
        # Para uma IA básica, score * 100 + lifespan funciona como ponto de partida.
        return self.score * 100 + self.lifespan