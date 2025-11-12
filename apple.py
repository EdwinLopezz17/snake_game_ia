#apple.py
from pygame.math import Vector2
import random
from snake import Snake

from constants import ANCHO, ALTO, SIZE, RED, GREEN, BLACK, WHITE


class Apple:
    def __init__(self, snake):
        self.pos = Vector2(0, 0)
        self.generate(snake)

    def get_available_spots(self, snake: Snake):
        occupied_spots = set() 

        for block in snake.body:
            occupied_spots.add((block.x, block.y))

        available_spots = []

        grid_width = int(ANCHO / SIZE)
        grid_height = int(ALTO / SIZE)

        for row in range(grid_width):
            for col in range(grid_height):
                potential_pos = Vector2(row * SIZE, col * SIZE)

                if (potential_pos.x, potential_pos.y) not in occupied_spots:
                    available_spots.append(potential_pos)

        return available_spots


    def generate(self, snake):
        available_spots = self.get_available_spots(snake)

        if available_spots:
            self.pos = random.choice(available_spots)
        else:
            print("¡Juego completado! No hay más espacio.")
            pass


    def draw(self, pygame, WIN):
        pygame.draw.rect(WIN,RED,(self.pos.x, self.pos.y, SIZE, SIZE))

