import pygame
import time
import numpy as np
from snake import Snake
from food import Food
from neural_network import NeuralNetwork
from config import LARGURA_TELA, ALTURA_TELA, PRETO, VERMELHO, BRANCO, CIMA, BAIXO, ESQUERDA, DIREITA, OUTPUT_NEURONS

class Game:
    def __init__(self, use_ai=True, nn_model=None, speed=15):
        pygame.init()
        self.screen = pygame.display.set_mode((LARGURA_TELA, ALTURA_TELA))
        pygame.display.set_caption('Jogo da Cobrinha IA')
        self.clock = pygame.time.Clock()
        self.font_style = pygame.font.SysFont("bahnschrift", 25)
        self.score_font = pygame.font.SysFont("comicsansms", 35)

        self.snake = Snake()
        self.food = Food()
        self.food.spawn(self.snake.get_body_pos()) 

        self.use_ai = use_ai
        self.nn_model = nn_model 
        self.speed = speed
        self.game_over = False 

    def _draw_score(self):
        value = self.score_font.render("Pontuação: " + str(self.snake.score), True, BRANCO)
        self.screen.blit(value, [0, 0])
        # Opcional: exibir tempo de vida/fitness da IA
        if self.use_ai:
             lifespan_text = self.font_style.render(f"Vida: {self.snake.lifespan}", True, BRANCO)
             self.screen.blit(lifespan_text, [0, 40])


    def _message(self, msg, color):
        renderizacao_msg = self.font_style.render(msg, True, color)
        # Centraliza a mensagem
        text_rect = renderizacao_msg.get_rect(center=(LARGURA_TELA / 2, ALTURA_TELA / 2))
        self.screen.blit(renderizacao_msg, text_rect)
        pygame.display.update()

    def _game_over_screen(self):
        self.screen.fill(PRETO)
        self._message(f"Game Over! Pontuação: {self.snake.score}", VERMELHO)

        # Espera um pouco antes de mostrar a opção de reiniciar/sair
        start_time = time.time()
        while time.time() - start_time < 3: 
             for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "quit"

        self._message("Pressione C para Jogar Novamente ou Q para Sair", BRANCO)

        while True: # Novo loop para a tela de Game Over
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "quit" # Sair do jogo
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        return "quit" # Sair do jogo
                    if event.key == pygame.K_c:
                        return "restart" # Reiniciar o jogo

    def _get_ai_decision(self):
        # Obtém o estado atual do jogo para a rede neural
        inputs = self.snake.get_state_for_nn(self.food.get_pos())

        # Passa as entradas pela rede neural
        outputs = self.nn_model.forward(inputs) 

        # Interpreta a saída para decidir o movimento
        # Encontra o índice da saída com o maior valor
        decision_index = np.argmax(outputs[0])

        # Mapeia o índice para uma ação: 0=Reto, 1=Esquerda, 2=Direita
        current_direction = self.snake.direction

        if decision_index == 0:
            new_direction = current_direction
        elif decision_index == 1:
            if current_direction == CIMA: new_direction = ESQUERDA
            elif current_direction == BAIXO: new_direction = DIREITA
            elif current_direction == ESQUERDA: new_direction = BAIXO
            elif current_direction == DIREITA: new_direction = CIMA
            else: new_direction = current_direction
        elif decision_index == 2:
            if current_direction == CIMA: new_direction = DIREITA
            elif current_direction == BAIXO: new_direction = ESQUERDA
            elif current_direction == ESQUERDA: new_direction = CIMA
            elif current_direction == DIREITA: new_direction = BAIXO
            else: new_direction = current_direction 
        else: 
             new_direction = current_direction
             print(f"AVISO: Saída inesperada da NN: {outputs}")


        self.snake.change_direction(new_direction)


    def run(self):
        game_exit = False

        while not game_exit:
            if self.snake.is_alive:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        game_exit = True
                        #Controle Manual
                    if not self.use_ai and event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_LEFT:
                            self.snake.change_direction(ESQUERDA)
                        elif event.key == pygame.K_RIGHT:
                            self.snake.change_direction(DIREITA)
                        elif event.key == pygame.K_UP:
                            self.snake.change_direction(CIMA)
                        elif event.key == pygame.K_DOWN:
                            self.snake.change_direction(BAIXO)
                if self.use_ai and self.nn_model:
                    self._get_ai_decision()
                self.snake.move()

                if self.snake.check_collision():
                    pass
                if self.snake.get_head_pos() == self.food.get_pos():
                    self.food.spawn(self.snake.get_body_pos())
                    self.snake.ate_comida() 
                self.screen.fill(PRETO) 
                self.food.draw(self.screen) 
                self.snake.draw(self.screen) 
                self._draw_score()
                pygame.display.update() 
                self.clock.tick(self.speed) 

            else:
                action = self._game_over_screen()
                if action == "quit":
                    game_exit = True
                elif action == "restart":
                    self.__init__(use_ai=self.use_ai, nn_model=self.nn_model, speed=self.speed)
                    pass

        pygame.quit() 
        quit() 
