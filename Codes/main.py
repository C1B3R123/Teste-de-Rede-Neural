# Arquivo: main.py

# Este é o ponto de entrada principal do seu programa.
# Ele decide se vai rodar o jogo em modo de exibição (com ou sem IA)
# ou se vai iniciar o processo de treinamento genético.

import pygame
from game import Game
from neural_network import NeuralNetwork
from config import INPUT_NEURONS, HIDDEN_NEURONS, OUTPUT_NEURONS, BEST_NN_FILE, VELOCIDADE_IA_DISPLAY
# from genetic_algorithm import GeneticAlgorithmManager # Descomente quando implementar o AG

if __name__ == "__main__":
    # --- Opção 1: Rodar o Jogo com a IA (carregando a melhor rede salva) ---

    # 1. Cria uma instância da Rede Neural
    nn = NeuralNetwork(INPUT_NEURONS, HIDDEN_NEURONS, OUTPUT_NEURONS)
    nn.load(BEST_NN_FILE) # Tenta carregar a rede "treinada" (se existir)

    # 2. Cria uma instância do Jogo, passando use_ai=True e o modelo da Rede Neural
    game = Game(use_ai=True, nn_model=nn, speed=VELOCIDADE_IA_DISPLAY)

    # 3. Executa o jogo
    game.run()

    # --- Opção 2: Rodar o Jogo com Controle Humano ---
    # Descomente as linhas abaixo e comente a Opção 1 para jogar manualmente

    # game_human = Game(use_ai=False, speed=15) # Defina a velocidade para o controle humano
    # game_human.run()

    # --- Opção 3: Iniciar o Treinamento Genético ---
    # Descomente as linhas abaixo e comente as Opções 1 e 2 para iniciar o treinamento
    # Lembre-se de descomentar a importação de GeneticAlgorithmManager acima
    # e completar a implementação da classe GeneticAlgorithmManager em genetic_algorithm.py

    # POPULATION_SIZE = 50
    # MUTATION_RATE = 0.1
    # NUM_GENERATIONS = 1000 # Quantas gerações treinar

    # ga_manager = GeneticAlgorithmManager(POPULATION_SIZE, MUTATION_RATE)

    # for generation in range(NUM_GENERATIONS):
    #     print(f"Iniciando Geração {generation + 1}")
    #     ga_manager.run_generation()
    #     # O método run_generation dentro do GA Manager deve simular os jogos
    #     # e atualizar a população para a próxima geração.
    #     # A melhor rede de cada geração pode ser salva automaticamente.