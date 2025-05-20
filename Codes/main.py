import pygame
import sys # For sys.exit()
from game import Game
from neural_network import NeuralNetwork
from genetic_algorithm import GeneticAlgorithmManager # Now importing the GA manager
from config import INPUT_NEURONS, HIDDEN_NEURONS, OUTPUT_NEURONS, BEST_NN_FILE, BLACK, WHITE

def display_menu(screen, font_style):
    screen.fill(BLACK)
    title = font_style.render("Jogo da Cobrinha IA", True, WHITE)
    option1 = font_style.render("1. Ver IA Treinando", True, WHITE)
    option2 = font_style.render("2. Jogar Sozinho", True, WHITE)
    option3 = font_style.render("3. Jogar Contra IA", True, WHITE)
    option4 = font_style.render("4. Treinar IA (Algoritmo Genético)", True, WHITE) # New option
    option5 = font_style.render("5. Sair", True, WHITE) # Adjusted option

    screen.blit(title, title.get_rect(center=(screen.get_width() / 2, screen.get_height() / 2 - 120)))
    screen.blit(option1, option1.get_rect(center=(screen.get_width() / 2, screen.get_height() / 2 - 50)))
    screen.blit(option2, option2.get_rect(center=(screen.get_width() / 2, screen.get_height() / 2 - 10)))
    screen.blit(option3, option3.get_rect(center=(screen.get_width() / 2, screen.get_height() / 2 + 30)))
    screen.blit(option4, option4.get_rect(center=(screen.get_width() / 2, screen.get_height() / 2 + 70)))
    screen.blit(option5, option5.get_rect(center=(screen.get_width() / 2, screen.get_height() / 2 + 110)))
    pygame.display.update()

def main():
    pygame.init()
    screen_width = 600
    screen_height = 400
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Menu Principal")
    font_style = pygame.font.SysFont("bahnschrift", 30)

    # Load the neural network once for modes 1 and 3
    nn_model_for_play = NeuralNetwork(INPUT_NEURONS, HIDDEN_NEURONS, OUTPUT_NEURONS)
    nn_model_for_play.load(BEST_NN_FILE) # Tries to load the "trained" network (if it exists)

    running = True
    while running:
        display_menu(screen, font_style)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    # Option 1: Watch AI Playing
                    game = Game(mode='ai_watch', nn_model=nn_model_for_play)
                    game.run()
                    # After game, re-init pygame to avoid display issues from closing/reopening
                    pygame.quit()
                    pygame.init()
                    screen = pygame.display.set_mode((screen_width, screen_height))
                    pygame.display.set_caption("Menu Principal")
                    font_style = pygame.font.SysFont("bahnschrift", 30)

                elif event.key == pygame.K_2:
                    # Option 2: Play Alone
                    game = Game(mode='human')
                    game.run()
                    pygame.quit()
                    pygame.init()
                    screen = pygame.display.set_mode((screen_width, screen_height))
                    pygame.display.set_caption("Menu Principal")
                    font_style = pygame.font.SysFont("bahnschrift", 30)

                elif event.key == pygame.K_3:
                    # Option 3: Play Against AI
                    game = Game(mode='human_vs_ai', nn_model=nn_model_for_play)
                    game.run()
                    pygame.quit()
                    pygame.init()
                    screen = pygame.display.set_mode((screen_width, screen_height))
                    pygame.display.set_caption("Menu Principal")
                    font_style = pygame.font.SysFont("bahnschrift", 30)

                elif event.key == pygame.K_4:
                    # Option 4: Train AI (Genetic Algorithm)
                    pygame.display.set_caption("Treinamento da IA (Algoritmo Genético)")
                    screen.fill(BLACK)
                    training_msg = font_style.render("Iniciando Treinamento da IA...", True, WHITE)
                    screen.blit(training_msg, training_msg.get_rect(center=(screen.get_width() / 2, screen.get_height() / 2)))
                    pygame.display.update()

                    # Important: Quit Pygame display before starting headless training
                    pygame.quit()

                    ga_manager = GeneticAlgorithmManager()
                    NUM_GENERATIONS = 100 # Example: Train for 100 generations

                    for generation_num in range(NUM_GENERATIONS):
                        ga_manager.run_generation()

                    print("\nTreinamento Concluído!")
                    # After training, re-initialize Pygame for the menu
                    pygame.init()
                    screen = pygame.display.set_mode((screen_width, screen_height))
                    pygame.display.set_caption("Menu Principal")
                    font_style = pygame.font.SysFont("bahnschrift", 30)
                    # Reload the best NN model after training
                    nn_model_for_play.load(BEST_NN_FILE)

                elif event.key == pygame.K_5:
                    # Option 5: Exit
                    running = False

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
