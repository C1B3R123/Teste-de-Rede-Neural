import pygame
import numpy as np

TAMANHO_BLOCO = 20
LARGURA_TELA = 600
ALTURA_TELA = 400

PRETO = (0, 0, 0)
BRANCO = (255, 255, 255)
VERMELHO = (213, 50, 80)
VERDE = (0, 255, 0)
AZUL = (50, 153, 213)

CIMA = (0, -1)
BAIXO = (0, 1)
ESQUERDA = (-1, 0)
DIREITA = (1, 0)

DIRECOES_POSSIVEIS = [CIMA, BAIXO, ESQUERDA, DIREITA]

# --- Configurações da Rede Neural ---
INPUT_NEURONS = 11 
HIDDEN_NEURONS = 16
OUTPUT_NEURONS = 3

# Nome do arquivo para salvar/carregar a melhor rede neural
BEST_NN_FILE = "best_gen_snk.pkl"

# Velocidade de exibição para a IA (FPS)
VELOCIDADE_IA_DISPLAY = 60
