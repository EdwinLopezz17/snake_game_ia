#apple.py
from pygame.math import Vector2
import random
from snake import Snake
from constants import ANCHO, ALTO, SIZE, RED

class Apple:
    def __init__(self, snake):
        self.pos = Vector2(0, 0)
        # Pre-calculate grid dimensions
        self.grid_width = int(ANCHO / SIZE)
        self.grid_height = int(ALTO / SIZE)
        self.generate(snake)

    def get_available_spots(self, snake: Snake):
        """
        OPTIMIZED: Uses set operations for faster collision detection.
        """
        # Create set of occupied positions (O(n) where n = snake length)
        occupied_spots = {(block.x, block.y) for block in snake.body}

        available_spots = []
        
        # Generate all possible positions
        for row in range(self.grid_width):
            for col in range(self.grid_height):
                x, y = row * SIZE, col * SIZE
                
                # Fast O(1) lookup in set
                if (x, y) not in occupied_spots:
                    available_spots.append(Vector2(x, y))

        return available_spots

    def generate(self, snake):
        """Generate apple in random available position."""
        available_spots = self.get_available_spots(snake)

        if available_spots:
            self.pos = random.choice(available_spots)
        else:
            print("¡Juego completado! No hay más espacio.")

    def draw(self, pygame, WIN):
        """Draw apple on screen."""
        pygame.draw.rect(WIN, RED, (self.pos.x, self.pos.y, SIZE, SIZE))