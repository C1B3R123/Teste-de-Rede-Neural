import pygame
from game import Game
from neural_network import NeuralNetwork
from config import INPUT_NEURONS, HIDDEN_NEURONS, OUTPUT_NEURONS, BEST_NN_FILE, VELOCIDADE_IA_DISPLAY
from genetic_algorithm import GeneticAlgorithmManager

if __name__ == "__main__":
    game_human = Game(use_ai=False, speed=15) 
    game_human.run()

   
