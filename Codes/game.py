import pygame
import time
import numpy as np
# Importa classes e configurações de outros arquivos
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
        self.food.spawn(self.snake.get_body_pos()) # Spawna a primeira comida

        self.use_ai = use_ai
        self.nn_model = nn_model # A rede neural que controlará a cobra (se use_ai=True)
        self.speed = speed # Velocidade do jogo (FPS)
        self.game_over = False # Adicionado para controle do loop principal

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
        while time.time() - start_time < 3: # Espera 3 segundos
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
        outputs = self.nn_model.forward(inputs) # outputs é um array numpy de shape (1, 3)

        # Interpreta a saída para decidir o movimento
        # Encontra o índice da saída com o maior valor
        decision_index = np.argmax(outputs[0]) # outputs[0] é a única linha (1D array)

        # Mapeia o índice para uma ação: 0=Reto, 1=Esquerda, 2=Direita
        # Calcula a nova direção baseada na decisão e direção atual
        current_direction = self.snake.direction

        if decision_index == 0: # Seguir Reto
            new_direction = current_direction
        elif decision_index == 1: # Virar para a Esquerda (90 graus relativo)
            if current_direction == CIMA: new_direction = ESQUERDA
            elif current_direction == BAIXO: new_direction = DIREITA
            elif current_direction == ESQUERDA: new_direction = BAIXO
            elif current_direction == DIREITA: new_direction = CIMA
            else: new_direction = current_direction # Caso inicial ou inesperado
        elif decision_index == 2: # Virar para a Direita (90 graus relativo)
            if current_direction == CIMA: new_direction = DIREITA
            elif current_direction == BAIXO: new_direction = ESQUERDA
            elif current_direction == ESQUERDA: new_direction = CIMA
            elif current_direction == DIREITA: new_direction = BAIXO
            else: new_direction = current_direction # Caso inicial ou inesperado
        else: # Caso inesperado, manter direção (não deve acontecer com argmax)
             new_direction = current_direction
             print(f"AVISO: Saída inesperada da NN: {outputs}")


        self.snake.change_direction(new_direction)


    def run(self):
        game_exit = False

        while not game_exit:
            if self.snake.is_alive: # Loop de jogo enquanto a cobra está viva
                # --- Eventos (apenas para controle humano ou para sair) ---
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        game_exit = True
                    if not self.use_ai and event.type == pygame.KEYDOWN: # Captura teclas APENAS se não estiver usando IA
                        if event.key == pygame.K_LEFT:
                            self.snake.change_direction(ESQUERDA)
                        elif event.key == pygame.K_RIGHT:
                            self.snake.change_direction(DIREITA)
                        elif event.key == pygame.K_UP:
                            self.snake.change_direction(CIMA)
                        elif event.key == pygame.K_DOWN:
                            self.snake.change_direction(BAIXO)

                # --- Lógica da IA ---
                if self.use_ai and self.nn_model: # Se usando IA e tem um modelo
                    self._get_ai_decision()

                # --- Movimentação e Colisões ---
                self.snake.move()

                if self.snake.check_collision():
                    # O is_alive da cobra já foi setado para False na colisão
                    pass # O loop principal vai para a tela de Game Over

                # Verifica se comeu a comida
                if self.snake.get_head_pos() == self.food.get_pos():
                    self.food.spawn(self.snake.get_body_pos()) # Nova comida
                    self.snake.ate_comida() # Cobra cresce
                    # A pontuação já é atualizada dentro do método ate_comida da cobra

                # --- Desenhar na Tela ---
                self.screen.fill(PRETO) # Fundo preto
                self.food.draw(self.screen) # Desenha comida
                self.snake.draw(self.screen) # Desenha cobra
                self._draw_score() # Desenha pontuação

                pygame.display.update() # Atualiza a tela

                # --- Controle de Velocidade ---
                self.clock.tick(self.speed) # Controla o FPS/velocidade

            else: # Se a cobra não está mais viva
                action = self._game_over_screen()
                if action == "quit":
                    game_exit = True
                elif action == "restart":
                    # Reinicia o jogo criando uma nova instância da classe Game
                    # Mantém a mesma configuração de IA/velocidade/NN
                    self.__init__(use_ai=self.use_ai, nn_model=self.nn_model, speed=self.speed)
                    # Não precisamos de uma flag de reinício aqui, o loop while not game_exit
                    # continuará com a nova instância do jogo.
                    pass # Continua no loop externo com a nova instância

        pygame.quit() # Sai do pygame
        # quit() # Deixamos o script Python sair naturalmente