Arquivo: genetic_algorithm.py

import numpy as np
from neural_network import NeuralNetwork
from game import Game
from config import INPUT_NEURONS, HIDDEN_NEURONS, OUTPUT_NEURONS, BEST_NN_FILE
class GeneticAlgorithmManager:
  def __init__(self, population_size, mutation_rate, ...):
    self.population_size = population_size
    self.mutation_rate = mutation_rate
        self.population = [NeuralNetwork(INPUT_NEURONS, HIDDEN_NEURONS, OUTPUT_NEURONS) for _ in range(population_size)]
        self.best_fitness_ever = 0
        self.generation = 0

    def run_generation(self):
        # Simula o jogo para cada cobra na população
        fitness_scores = []
        for i, nn_model in enumerate(self.population):
            # Rode o jogo (Game) usando este nn_model
            # Em um AG, você rodaria o jogo "sem interface gráfica" para ser rápido
            game_sim = Game(use_ai=True, nn_model=nn_model, show_gui=False, speed=alta)
            game_sim.run()
            fitness = game_sim.snake.get_fitness()
            fitness_scores.append((fitness, nn_model))
            pass 

        # Ordena por fitness (do melhor para o pior)
        fitness_scores.sort(key=lambda x: x[0], reverse=True)

        # Salva a melhor rede neural desta geração (se for a melhor já vista)
        best_gen_snk = fitness_scores[0][1]
        if fitness_scores[0][0] > self.best_fitness_ever:
          self.best_fitness_ever = fitness_scores[0][0]
          best_nn_gen.save(BEST_NN_FILE)
          print(f"Nova melhor fitness: {self.best_fitness_ever} (Geração {self.generation})")

        Seleção (escolhe os pais com base no fitness)
        parents = self._select(fitness_scores) # TODO: Implementar seleção

        # Cria a próxima geração (crossover e mutação)
        next_population = self._create_next_generation(parents) # TODO: Implementar crossover/mutação

        self.population = next_population
        self.generation += 1
