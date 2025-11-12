#game.py
import pygame
from pygame.math import Vector2
import random

from snake import Snake
from apple import Apple

from constants import ANCHO, ALTO, SIZE, RED, GREEN, BLACK, WHITE, FPS


pygame.init()

class Game:
    def __init__(self):
        self.WIN = pygame.display.set_mode((ANCHO, ALTO))
        pygame.display.set_caption("Snake AI Ready")

        self.SCORE_TEXT = pygame.font.SysFont("Russo One", 15)
        self.TITLE_FONT = pygame.font.SysFont("Russo One", 30)
        self.MENU_FONT = pygame.font.SysFont("Russo One", 15)

        self.fps_clock = pygame.time.Clock()

        self._init_elements()

    def _init_elements(self):
        self.snake = Snake()
        self.apple = Apple(self.snake)
        self.score = 0
        
        self.game_active = True


    def _check_colision(self):
        if self.apple.pos == self.snake.body[0]:
            return True
        return False
    
    def _draw_elements(self):
        self.WIN.fill(BLACK)
        self.snake.draw(pygame, self.WIN)
        self.apple.draw(pygame, self.WIN)

        text = self.SCORE_TEXT.render(f"Score: {self.score}", 1, WHITE)
        self.WIN.blit(text, (ANCHO - text.get_width(), 0))

        pygame.display.update()

    def _handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._quit_game()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w: self.snake.move_up()
                elif event.key == pygame.K_s: self.snake.move_down()
                elif event.key == pygame.K_d: self.snake.move_right()
                elif event.key == pygame.K_a: self.snake.move_left()
    
    def _update_game_state(self):
        self.snake.can_change_direction = True
        self.snake.move()

        if self._check_colision():
            self.score += 1
            self.snake.add = True
            self.apple.generate(self.snake)

        if self.snake.die():
            self.game_active = False

    def _game_over_screen(self):
        title_text = self.TITLE_FONT.render("GAME OVER", 1, WHITE)
        score_text = self.SCORE_TEXT.render(f"Puntuación Final: {self.score}", 1, WHITE)
        restart_text = self.MENU_FONT.render("Presiona R para Reiniciar", 1, GREEN)
        quit_text = self.MENU_FONT.render("Presiona ESC o Q para Salir", 1, (255, 100, 100))

        self.WIN.fill(BLACK) 

        self.WIN.blit(title_text, (ANCHO//2 - title_text.get_width()//2, ALTO//2 - 50))
        self.WIN.blit(score_text, (ANCHO//2 - score_text.get_width()//2, ALTO//2 - 10))
        self.WIN.blit(restart_text, (ANCHO//2 - restart_text.get_width()//2, ALTO//2 + 30))
        self.WIN.blit(quit_text, (ANCHO//2 - quit_text.get_width()//2, ALTO//2 + 50))

        pygame.display.update()

        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._quit_game()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        return "RESTART"
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                        self._quit_game()
        return "QUIT"


    def _quit_game(self):
        pygame.quit()
        quit()

    def run(self):
        running = True
        
        while running:
            self.game_active = True
            
            while self.game_active:
                self.fps_clock.tick(FPS)

                self._handle_input()
                self._update_game_state()
                self._draw_elements()

            action = self._game_over_screen()

            if action == "RESTART":
                self._init_elements()
                continue
            else:
                running = False

if __name__ == '__main__':
    game = Game()
    game.run()


    