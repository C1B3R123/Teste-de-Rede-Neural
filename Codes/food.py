import pygame
import random
# Importa configurações do arquivo config.py
from config import TAMANHO_BLOCO, LARGURA_TELA, ALTURA_TELA, AZUL

class Food:
    def __init__(self):
        self.position = (0, 0)
        # A comida será spawnada pela classe Game ou GeneticAlgorithmManager

    def spawn(self, snake_body):
        while True:
            x = random.randrange(0, LARGURA_TELA // TAMANHO_BLOCO) * TAMANHO_BLOCO
            y = random.randrange(0, ALTURA_TELA // TAMANHO_BLOCO) * TAMANHO_BLOCO
            self.position = (x, y)
            # Garante que a comida não aparece DENTRO da cobra
            if self.position not in snake_body:
                break

    def draw(self, screen):
        pygame.draw.rect(screen, AZUL, [self.position[0], self.position[1], TAMANHO_BLOCO, TAMANHO_BLOCO])

    def get_pos(self):
        return self.position