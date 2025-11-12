#snake.py
from pygame.math import Vector2
from constants import ANCHO, ALTO, SIZE, GREEN

class Snake:
    def __init__(self):
        self.body = [Vector2(SIZE*5,SIZE),Vector2(SIZE*4,SIZE),Vector2(SIZE*3,SIZE)]
        self.direction_up = Vector2(0,-SIZE)
        self.direction_down = Vector2(0,SIZE)
        self.direction_left = Vector2(-SIZE,0)
        self.direction_right = Vector2(SIZE,0)
        self.can_change_direction = True

        self.direction = self.direction_right
        self.add = False
    
    def draw(self, pygame, WIN):
        for block in self.body:
            pygame.draw.rect(WIN,GREEN,(block.x,block.y,SIZE,SIZE))

    def move(self):
        head = self.body[0] + self.direction
        self.body.insert(0,head)

        if not self.add:
            self.body.pop()
        else:
            self.add = False

    def move_up(self):
        if self.direction == self.direction_down or self.can_change_direction == False:
            return
        self.direction = self.direction_up
        self.can_change_direction = False
        
    def move_down(self):
        if self.direction == self.direction_up or self.can_change_direction == False: 
            return
        self.direction = self.direction_down
        self.can_change_direction = False
        

    def move_left(self):
        if self.direction == self.direction_right or self.can_change_direction == False: 
            return
        self.direction = self.direction_left
        self.can_change_direction = False

    def move_right(self):
        if self.direction == self.direction_left or self.can_change_direction == False:
            return
        self.direction = self.direction_right
        self.can_change_direction = False

    def die (self):
        if (self.body[0].x < 0 or self.body[0].x >= ANCHO or self.body[0].y < 0 or self.body[0].y >= ALTO):
            return True
        
        for block in self.body[1:]:
            if self.body[0] == block:
                return True
        
        return False
   