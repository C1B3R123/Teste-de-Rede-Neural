# File: genetic_algorithm.py

import numpy as np
import random
from neural_network import NeuralNetwork
from game import Game
from config import (INPUT_NEURONS, HIDDEN_NEURONS, OUTPUT_NEURONS, BEST_NN_FILE,
                    POPULATION_SIZE, MUTATION_RATE, MUTATION_STRENGTH, ELITISM_COUNT)

class GeneticAlgorithmManager:
    def __init__(self, population_size=POPULATION_SIZE, mutation_rate=MUTATION_RATE,
                 mutation_strength=MUTATION_STRENGTH, elitism_count=ELITISM_COUNT):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.elitism_count = elitism_count

        self.population = [NeuralNetwork(INPUT_NEURONS, HIDDEN_NEURONS, OUTPUT_NEURONS) for _ in range(population_size)]
        self.best_fitness_ever = -float('inf') # Initialize with a very low number
        self.generation = 0

    def run_generation(self):
        self.generation += 1
        print(f"\n--- Generation {self.generation} ---")

        fitness_scores = []
        for i, nn_model in enumerate(self.population):
            # Run the game for this snake in headless mode to calculate fitness
            # Each game instance needs its own Pygame initialization if not headless,
            # but in headless mode, it won't create a display.
            # We set headless=True here for training speed.
            game_sim = Game(mode='ai_watch', nn_model=nn_model, headless=True)
            game_sim.run() # This runs the game loop until game_over
            fitness = game_sim.ai_snake.get_fitness()
            fitness_scores.append((fitness, nn_model))
            # print(f"  Snake {i+1} Fitness: {fitness}") # Optional: print individual fitness

        # Sort by fitness (best to worst)
        fitness_scores.sort(key=lambda x: x[0], reverse=True)

        # Print best fitness of this generation
        current_best_fitness = fitness_scores[0][0]
        print(f"Generation {self.generation} Best Fitness: {current_best_fitness}")

        # Save the best neural network of this generation (if it's the best ever seen)
        best_nn_gen = fitness_scores[0][1]
        if current_best_fitness > self.best_fitness_ever:
            self.best_fitness_ever = current_best_fitness
            best_nn_gen.save(BEST_NN_FILE)
            print(f"New global best fitness: {self.best_fitness_ever}")

        # Selection: Choose parents based on fitness (e.g., top N individuals)
        # Simple truncation selection: take the top N individuals as parents for the next generation
        parents = [nn for fitness, nn in fitness_scores[:self.population_size // 2]] # Take top 50% as parents

        # Create the next generation
        next_population = []

        # Elitism: Carry over the very best individuals directly to the next generation
        for i in range(min(self.elitism_count, len(fitness_scores))):
            next_population.append(fitness_scores[i][1])

        # Fill the rest of the population through crossover and mutation
        while len(next_population) < self.population_size:
            # Randomly select two parents (can be the same parent, or use a more sophisticated selection like roulette wheel)
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)

            # Crossover
            child_weights = self._crossover(parent1.get_weights(), parent2.get_weights())

            # Create a new NN for the child
            child_nn = NeuralNetwork(INPUT_NEURONS, HIDDEN_NEURONS, OUTPUT_NEURONS)
            child_nn.set_weights(child_weights)

            # Mutate
            self._mutate(child_nn)

            next_population.append(child_nn)

        self.population = next_population

    def _crossover(self, parent1_weights, parent2_weights):
        # Simple uniform crossover for weights and biases
        # For each weight matrix/bias vector, randomly choose from parent1 or parent2
        W1_p1, b1_p1, W2_p1, b2_p1 = parent1_weights
        W1_p2, b1_p2, W2_p2, b2_p2 = parent2_weights

        child_W1 = W1_p1 if random.random() < 0.5 else W1_p2
        child_b1 = b1_p1 if random.random() < 0.5 else b1_p2
        child_W2 = W2_p1 if random.random() < 0.5 else W2_p2
        child_b2 = b2_p1 if random.random() < 0.5 else b2_p2

        return (child_W1, child_b1, child_W2, child_b2)

    def _mutate(self, nn_model):
        # Mutate the weights and biases of the neural network
        W1, b1, W2, b2 = nn_model.get_weights()

        # Mutate W1
        for i in range(W1.shape[0]):
            for j in range(W1.shape[1]):
                if random.random() < self.mutation_rate:
                    W1[i, j] += np.random.randn() * self.mutation_strength

        # Mutate b1
        for i in range(b1.shape[1]):
            if random.random() < self.mutation_rate:
                b1[0, i] += np.random.randn() * self.mutation_strength

        # Mutate W2
        for i in range(W2.shape[0]):
            for j in range(W2.shape[1]):
                if random.random() < self.mutation_rate:
                    W2[i, j] += np.random.randn() * self.mutation_strength

        # Mutate b2
        for i in range(b2.shape[1]):
            if random.random() < self.mutation_rate:
                b2[0, i] += np.random.randn() * self.mutation_strength

        nn_model.set_weights((W1, b1, W2, b2))
