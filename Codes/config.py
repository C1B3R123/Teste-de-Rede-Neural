# File: config.py

import pygame
import numpy as np

# --- Game Settings ---
BLOCK_SIZE = 20
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 400

# Colors (RGB)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (213, 50, 80)
GREEN = (0, 255, 0)
BLUE = (50, 153, 213)
PURPLE = (128, 0, 128) # New color for the second snake

# Directions (as vectors for easier next position calculation)
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

POSSIBLE_DIRECTIONS = [UP, DOWN, LEFT, RIGHT]

# --- Neural Network Settings ---
INPUT_NEURONS = 11 # Number of inputs for the neural network (sensors + state)
HIDDEN_NEURONS = 16 # Number of neurons in the hidden layer
OUTPUT_NEURONS = 3 # Number of outputs It's a only 3 outputs neurons, so: Turn Left, Right or Go Straight

# File name to save/load the best neural network
BEST_NN_FILE = "best_snake_nn.pkl"

# Display speeds (FPS)
HUMAN_SPEED = 15
AI_DISPLAY_SPEED = 30
VS_AI_SPEED = 20 

# --- Genetic Algorithm Settings ---
POPULATION_SIZE = 20
MUTATION_RATE = 0.01
MUTATION_STRENGTH = 0.1
ELITISM_COUNT = 2
