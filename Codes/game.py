# File: game.py

import pygame
import time
import numpy as np
# Import classes and settings from other files
from snake import Snake
from food import Food
from neural_network import NeuralNetwork
from config import (SCREEN_WIDTH, SCREEN_HEIGHT, BLACK, RED, WHITE, GREEN, PURPLE, BLUE,
                    UP, DOWN, LEFT, RIGHT, OUTPUT_NEURONS,
                    HUMAN_SPEED, AI_DISPLAY_SPEED, VS_AI_SPEED)

class Game:
    def __init__(self, mode='human', nn_model=None, headless=False):
        self.mode = mode
        self.nn_model = nn_model # NN model for AI
        self.headless = headless # If True, no display will be created

        if not self.headless:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption('Jogo da Cobrinha IA')
            self.clock = pygame.time.Clock()
            self.font_style = pygame.font.SysFont("bahnschrift", 25)
            self.score_font = pygame.font.SysFont("comicsansms", 35)
        else:
            # In headless mode, we don't need Pygame's display or font modules
            self.screen = None
            self.clock = None
            self.font_style = None
            self.score_font = None

        # Configure snakes and speed based on mode
        if self.mode == 'human':
            self.human_snake = Snake((SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2), RIGHT, GREEN)
            self.current_speed = HUMAN_SPEED
            self.snakes = [self.human_snake]
        elif self.mode == 'ai_watch':
            if not self.nn_model:
                raise ValueError("Neural network model is required for 'ai_watch' mode.")
            # For GA training, AI snakes should start in random-ish spots to avoid bias
            if self.headless:
                start_x = np.random.randint(0, SCREEN_WIDTH // BLOCK_SIZE) * BLOCK_SIZE
                start_y = np.random.randint(0, SCREEN_HEIGHT // BLOCK_SIZE) * BLOCK_SIZE
                start_dir = np.random.choice([UP, DOWN, LEFT, RIGHT])
                self.ai_snake = Snake((start_x, start_y), start_dir, GREEN)
            else:
                self.ai_snake = Snake((SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2), RIGHT, GREEN)
            self.current_speed = AI_DISPLAY_SPEED
            self.snakes = [self.ai_snake]
        elif self.mode == 'human_vs_ai':
            if not self.nn_model:
                raise ValueError("Neural network model is required for 'human_vs_ai' mode.")
            self.human_snake = Snake((SCREEN_WIDTH // 4, SCREEN_HEIGHT // 2), RIGHT, GREEN)
            self.ai_snake = Snake((SCREEN_WIDTH * 3 // 4, SCREEN_HEIGHT // 2), LEFT, PURPLE) # AI starts on the other side
            self.current_speed = VS_AI_SPEED
            self.snakes = [self.human_snake, self.ai_snake]
        else:
            raise ValueError("Invalid game mode.")

        self.food = Food()
        self._spawn_food() # Spawn the first food

        self.game_over = False
        self.winner = None # For VS mode

        # Max moves without eating food to prevent infinite loops for AI
        # This is crucial for training, otherwise snakes can get stuck and run forever
        self.max_moves_without_food = 200 * (SCREEN_WIDTH * SCREEN_HEIGHT / (BLOCK_SIZE * BLOCK_SIZE)) / len(self.snakes)
        self.moves_since_last_food = 0


    def _spawn_food(self):
        all_occupied_positions = []
        for snake in self.snakes:
            all_occupied_positions.extend(snake.get_body_pos())
        self.food.spawn(all_occupied_positions)
        self.moves_since_last_food = 0 # Reset counter when food is spawned

    def _draw_score(self):
        if self.headless: return # Do not draw in headless mode

        if self.mode == 'human' or self.mode == 'ai_watch':
            snake = self.snakes[0]
            value = self.score_font.render("Pontuação: " + str(snake.score), True, WHITE)
            self.screen.blit(value, [0, 0])
            if self.mode == 'ai_watch':
                 lifespan_text = self.font_style.render(f"Vida: {snake.lifespan}", True, WHITE)
                 self.screen.blit(lifespan_text, [0, 40])
        elif self.mode == 'human_vs_ai':
            human_score_text = self.score_font.render(f"Humano: {self.human_snake.score}", True, GREEN)
            ai_score_text = self.score_font.render(f"IA: {self.ai_snake.score}", True, PURPLE)
            self.screen.blit(human_score_text, [0, 0])
            self.screen.blit(ai_score_text, [SCREEN_WIDTH - ai_score_text.get_width(), 0])
            # Display life status
            human_status_text = self.font_style.render(f"Humano Vivo: {self.human_snake.is_alive}", True, GREEN)
            ai_status_text = self.font_style.render(f"IA Viva: {self.ai_snake.is_alive}", True, PURPLE)
            self.screen.blit(human_status_text, [0, 40])
            self.screen.blit(ai_status_text, [SCREEN_WIDTH - ai_status_text.get_width(), 40])


    def _display_message(self, msg, color, y_offset=0):
        if self.headless: return # Do not display messages in headless mode
        message_render = self.font_style.render(msg, True, color)
        text_rect = message_render.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 + y_offset))
        self.screen.blit(message_render, text_rect)
        pygame.display.update()

    def _game_over_screen(self):
        if self.headless: return "quit" # In headless mode, just exit gracefully

        self.screen.fill(BLACK)
        if self.mode == 'human_vs_ai':
            if self.winner:
                self._display_message(f"Fim de Jogo! Vencedor: {self.winner}!", RED, -50)
            else:
                self._display_message("Fim de Jogo! Empate!", RED, -50)
        else:
            self._display_message(f"Fim de Jogo! Pontuação: {self.snakes[0].score}", RED, -50)

        self._display_message("Pressione C para Jogar Novamente ou Q para Sair", WHITE, 50)
        pygame.display.update() # Ensure the final message appears

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "quit"
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        return "quit"
                    if event.key == pygame.K_c:
                        return "restart"

    def _get_ai_decision(self, ai_snake, other_snake_body=None):
        inputs = ai_snake.get_state_for_nn(self.food.get_pos(), other_snake_body)
        outputs = self.nn_model.forward(inputs)
        decision_index = np.argmax(outputs[0])

        current_direction = ai_snake.direction
        new_direction = current_direction

        if decision_index == 0: # Go Straight
            new_direction = current_direction
        elif decision_index == 1: # Turn Left (90 degrees relative)
            if current_direction == UP: new_direction = LEFT
            elif current_direction == DOWN: new_direction = RIGHT
            elif current_direction == LEFT: new_direction = DOWN
            elif current_direction == RIGHT: new_direction = UP
        elif decision_index == 2: # Turn Right (90 degrees relative)
            if current_direction == UP: new_direction = RIGHT
            elif current_direction == DOWN: new_direction = LEFT
            elif current_direction == LEFT: new_direction = UP
            elif current_direction == RIGHT: new_direction = DOWN

        ai_snake.change_direction(new_direction)

    def run(self):
        game_exit = False

        while not game_exit:
            # --- Event Handling ---
            if not self.headless: # Only process events if not headless
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        game_exit = True
                    if self.mode != 'ai_watch' and event.type == pygame.KEYDOWN: # Only for modes with human control
                        if self.mode == 'human':
                            if event.key == pygame.K_LEFT: self.human_snake.change_direction(LEFT)
                            elif event.key == pygame.K_RIGHT: self.human_snake.change_direction(RIGHT)
                            elif event.key == pygame.K_UP: self.human_snake.change_direction(UP)
                            elif event.key == pygame.K_DOWN: self.human_snake.change_direction(DOWN)
                        elif self.mode == 'human_vs_ai' and self.human_snake.is_alive:
                            if event.key == pygame.K_LEFT: self.human_snake.change_direction(LEFT)
                            elif event.key == pygame.K_RIGHT: self.human_snake.change_direction(RIGHT)
                            elif event.key == pygame.K_UP: self.human_snake.change_direction(UP)
                            elif event.key == pygame.K_DOWN: self.human_snake.change_direction(DOWN)
            else: # If headless, check for external quit signals (e.g., from GA manager)
                # In a real headless setup, you might have a different way to break the loop
                # For now, we'll rely on game_over for the GA.
                pass


            # --- Game Logic (if not yet Game Over) ---
            if not self.game_over:
                # Update moves since last food
                self.moves_since_last_food += 1
                if self.moves_since_last_food > self.max_moves_without_food:
                    # Force game over if snake gets stuck or cannot find food
                    for snake in self.snakes:
                        snake.is_alive = False
                    self.game_over = True
                    self.winner = "Tempo Esgotado" # Or similar, if needed for VS mode

                # Movement and AI
                if self.mode == 'human':
                    self.human_snake.move()
                elif self.mode == 'ai_watch':
                    self._get_ai_decision(self.ai_snake)
                    self.ai_snake.move()
                elif self.mode == 'human_vs_ai':
                    if self.human_snake.is_alive:
                        self.human_snake.move()
                    if self.ai_snake.is_alive:
                        self._get_ai_decision(self.ai_snake, self.human_snake.get_body_pos()) # AI considers the other snake
                        self.ai_snake.move()

                # Collisions
                if self.mode == 'human':
                    if self.human_snake.check_collision():
                        self.game_over = True
                elif self.mode == 'ai_watch':
                    if self.ai_snake.check_collision():
                        self.game_over = True
                elif self.mode == 'human_vs_ai':
                    # Check collisions for human snake
                    if self.human_snake.is_alive:
                        if self.human_snake.check_collision(self.ai_snake.get_body_pos()): # Human vs wall/self/AI
                            self.game_over = True
                            if not self.human_snake.is_alive and self.ai_snake.is_alive:
                                self.winner = "IA"
                    # Check collisions for AI snake
                    if self.ai_snake.is_alive:
                        if self.ai_snake.check_collision(self.human_snake.get_body_pos()): # AI vs wall/self/Human
                            self.game_over = True
                            if not self.ai_snake.is_alive and self.human_snake.is_alive:
                                self.winner = "Humano"

                    # Check if both are dead (draw)
                    if not self.human_snake.is_alive and not self.ai_snake.is_alive:
                        self.game_over = True
                        self.winner = "Ninguém (Empate)"
                    # If one is dead and the other is alive, the game is over and winner is set above
                    elif (self.human_snake.is_alive and not self.ai_snake.is_alive):
                        self.game_over = True
                        self.winner = "Humano"
                    elif (not self.human_snake.is_alive and self.ai_snake.is_alive):
                        self.game_over = True
                        self.winner = "IA"


                # Food
                food_eaten = False
                if self.mode == 'human' and self.food.get_pos() == self.human_snake.get_head_pos() and self.human_snake.is_alive:
                    self.human_snake.ate_food()
                    food_eaten = True
                elif self.mode == 'ai_watch' and self.food.get_pos() == self.ai_snake.get_head_pos() and self.ai_snake.is_alive:
                    self.ai_snake.ate_food()
                    food_eaten = True
                elif self.mode == 'human_vs_ai':
                    if self.food.get_pos() == self.human_snake.get_head_pos() and self.human_snake.is_alive:
                        self.human_snake.ate_food()
                        food_eaten = True
                    elif self.food.get_pos() == self.ai_snake.get_head_pos() and self.ai_snake.is_alive:
                        self.ai_snake.ate_food()
                        food_eaten = True

                if food_eaten:
                    self._spawn_food()


                # Drawing (only if not headless)
                if not self.headless:
                    self.screen.fill(BLACK)
                    self.food.draw(self.screen)
                    for snake in self.snakes:
                        snake.draw(self.screen)
                    self._draw_score()
                    pygame.display.update()

                    self.clock.tick(self.current_speed)
                else: # Headless mode - no drawing, just advance logic
                    # You might add a small delay here if computation is too fast for monitoring
                    # time.sleep(0.001)
                    pass

                # Check if all snakes are dead in multi-snake modes
                if self.mode == 'human_vs_ai':
                    if not self.human_snake.is_alive and not self.ai_snake.is_alive:
                        self.game_over = True
                elif self.mode == 'human' and not self.human_snake.is_alive:
                     self.game_over = True
                elif self.mode == 'ai_watch' and not self.ai_snake.is_alive:
                     self.game_over = True

            else: # If game_over is True
                if not self.headless:
                    action = self._game_over_screen()
                    if action == "quit":
                        game_exit = True
                    elif action == "restart":
                        # Restart the game with the same mode
                        self.__init__(mode=self.mode, nn_model=self.nn_model, headless=self.headless)
                        self.game_over = False # Reset game_over for the new game
                else: # Headless mode, just exit the loop
                    game_exit = True

        if not self.headless:
            pygame.quit()
        # No sys.exit() here to allow the main menu to continue
