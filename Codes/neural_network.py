import numpy as np
import pickle
import os
from config import INPUT_NEURONS, HIDDEN_NEURONS, OUTPUT_NEURONS

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.b2 = np.zeros((1, self.output_size))

    def forward(self, inputs):
        z1 = np.dot(inputs, self.W1) + self.b1
        a1 = np.maximum(0, z1) 
        z2 = np.dot(a1, self.W2) + self.b2
        return z2

    def get_weights(self):
        return (self.W1, self.b1, self.W2, self.b2)

    def set_weights(self, weights):
        W1, b1, W2, b2 = weights
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2

    def save(self, filename):
        # Salva os pesos da rede em um arquivo
        with open(filename, 'wb') as f:
            pickle.dump(self.get_weights(), f)
        print(f"Rede neural salva em {filename}")

    def load(self, filename):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                weights = pickle.load(f)
            self.set_weights(weights)
            print(f"Rede neural carregada de {filename}")
            return True
        else:
            print(f"Arquivo {filename} não encontrado.")
            return False
