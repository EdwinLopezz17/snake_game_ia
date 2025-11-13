#game.py
import pygame
from pygame.math import Vector2
import numpy as np

from snake import Snake
from apple import Apple
from dqn_agent import DQNAgent

from constants import ANCHO, ALTO, SIZE, RED, GREEN, BLACK, WHITE, FPS

# Rewards for the agent
REWARD_EAT = 10
REWARD_MOVE_CLOSER = 0.2
REWARD_MOVE_AWAY = -0.15
REWARD_DEATH = -10
REWARD_LOOP = -5  # Penalty for going in circles

# STATE SIZE: Enhanced spatial awareness
# 8 (danger in 8 directions) + 
# 8 (body parts in 8 directions, 2 steps away) + 
# 4 (current direction) + 
# 4 (apple location) + 
# 4 (snake info: length, body density, wall proximity, tail accessible)
STATE_SIZE = 28
ACTION_SIZE = 4

pygame.init()

class Game:
    def __init__(self, is_ai_mode=True):
        self.is_ai_mode = is_ai_mode
        
        self.WIN = pygame.display.set_mode((ANCHO, ALTO))
        pygame.display.set_caption("Snake AI V2 - Enhanced Spatial Vision")

        self.SCORE_TEXT = pygame.font.SysFont("Russo One", 12)
        self.TITLE_FONT = pygame.font.SysFont("Russo One", 30)
        self.MENU_FONT = pygame.font.SysFont("Russo One", 15)

        self.fps_clock = pygame.time.Clock()

        if self.is_ai_mode:
            self.agent = DQNAgent(STATE_SIZE, ACTION_SIZE)
        
        self._init_elements()
        
        self.episode_count = 0
        self.total_score = 0
        self.max_score = 0
        self.save_frequency = 2
        
        # Track recent positions to detect loops
        self.recent_positions = []

    def _init_elements(self):
        """Initialize game elements."""
        self.snake = Snake()
        self.apple = Apple(self.snake)
        self.score = 0
        self.game_active = True
        self.frame_iteration = 0
        self.prev_distance = self._get_distance_to_apple()
        self.recent_positions = []

    def _get_distance_to_apple(self):
        head = self.snake.body[0]
        return abs(head.x - self.apple.pos.x) + abs(head.y - self.apple.pos.y)

    def _check_colision(self):
        return self.apple.pos == self.snake.body[0]

    def _draw_elements(self):
        self.WIN.fill(BLACK)
        self.snake.draw(pygame, self.WIN)
        self.apple.draw(pygame, self.WIN)

        if self.is_ai_mode:
            info_lines = [
                f"Episode: {self.episode_count} | Score: {self.score} | Max: {self.max_score}",
                f"Length: {len(self.snake.body)} | Epsilon: {self.agent.epsilon:.3f}"
            ]
            
            for i, line in enumerate(info_lines):
                text = self.SCORE_TEXT.render(line, 1, WHITE)
                self.WIN.blit(text, (5, 5 + i * 15))
        else:
            score_text = self.SCORE_TEXT.render(
                f"Score: {self.score} | Length: {len(self.snake.body)}", 
                1, WHITE
            )
            self.WIN.blit(score_text, (5, 5))
            
        pygame.display.update()

    def _handle_input(self):
        """Handle human input if not in AI mode."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._quit_game()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w: self.snake.move_up()
                elif event.key == pygame.K_s: self.snake.move_down()
                elif event.key == pygame.K_d: self.snake.move_right()
                elif event.key == pygame.K_a: self.snake.move_left()

    def _is_collision_at(self, point):
        """
        Check if a point would cause collision.
        Returns: 1 if collision, 0 if safe
        """
        # Wall collision
        if point.x < 0 or point.x >= ANCHO or point.y < 0 or point.y >= ALTO:
            return 1
        # Body collision (skip head)
        if point in self.snake.body[1:]:
            return 1
        return 0

    def _count_body_parts_at(self, point, max_distance=2):
        """
        Count how many body parts are near a point.
        Returns normalized count (0 to 1).
        """
        count = 0
        for block in self.snake.body[1:]:
            distance = abs(block.x - point.x) + abs(block.y - point.y)
            if distance <= max_distance * SIZE:
                count += 1
        
        # Normalize by snake length
        return min(count / max(len(self.snake.body) - 1, 1), 1.0)

    def _get_state(self):
        """
        Generate enhanced state vector (28 values) for Neural Network.
        
        State breakdown:
        [0-7]:   Immediate danger in 8 directions (N, NE, E, SE, S, SW, W, NW)
        [8-15]:  Body density in 8 directions (how many body parts nearby)
        [16-19]: Current direction (L, R, U, D)
        [20-23]: Apple location (L, R, U, D)
        [24]:    Normalized snake length (0-1)
        [25]:    Body density around head (how trapped we are)
        [26]:    Distance to nearest wall (normalized)
        [27]:    Can reach tail (escape route available)
        """
        head = self.snake.body[0]
        
        dir_l = self.snake.direction_left
        dir_r = self.snake.direction_right
        dir_u = self.snake.direction_up
        dir_d = self.snake.direction_down
        
        # 8 directions: N, NE, E, SE, S, SW, W, NW
        directions_8 = [
            Vector2(0, -SIZE),           # N (0)
            Vector2(SIZE, -SIZE),        # NE (1)
            Vector2(SIZE, 0),            # E (2)
            Vector2(SIZE, SIZE),         # SE (3)
            Vector2(0, SIZE),            # S (4)
            Vector2(-SIZE, SIZE),        # SW (5)
            Vector2(-SIZE, 0),           # W (6)
            Vector2(-SIZE, -SIZE)        # NW (7)
        ]
        
        # [0-7] Immediate danger in 8 directions
        danger_8_dirs = []
        for direction in directions_8:
            point = Vector2(head.x + direction.x, head.y + direction.y)
            danger_8_dirs.append(self._is_collision_at(point))
        
        # [8-15] Body density in 8 directions (2 steps away)
        body_density_8_dirs = []
        for direction in directions_8:
            point = Vector2(head.x + direction.x * 2, head.y + direction.y * 2)
            density = self._count_body_parts_at(point, max_distance=3)
            body_density_8_dirs.append(density)
        
        # [16-19] Current direction
        current_direction = [
            int(self.snake.direction == dir_l),
            int(self.snake.direction == dir_r),
            int(self.snake.direction == dir_u),
            int(self.snake.direction == dir_d)
        ]
        
        # [20-23] Apple location relative to head
        apple_location = [
            int(self.apple.pos.x < head.x),  # Apple left
            int(self.apple.pos.x > head.x),  # Apple right
            int(self.apple.pos.y < head.y),  # Apple up
            int(self.apple.pos.y > head.y)   # Apple down
        ]
        
        # [24] Normalized snake length
        max_possible_length = (ANCHO // SIZE) * (ALTO // SIZE)
        normalized_length = len(self.snake.body) / max_possible_length
        
        # [25] Body density immediately around head (3x3 grid)
        body_count_around = 0
        for dx in [-SIZE, 0, SIZE]:
            for dy in [-SIZE, 0, SIZE]:
                if dx == 0 and dy == 0:
                    continue
                check_point = Vector2(head.x + dx, head.y + dy)
                if check_point in self.snake.body[1:]:
                    body_count_around += 1
        body_density_immediate = body_count_around / 8
        
        # [26] Distance to nearest wall (normalized 0-1)
        dist_to_walls = [
            head.x / ANCHO,                    # From left
            (ANCHO - head.x) / ANCHO,         # From right
            head.y / ALTO,                     # From top
            (ALTO - head.y) / ALTO            # From bottom
        ]
        min_wall_distance = min(dist_to_walls)
        
        # [27] Can reach tail? (Check if path to tail is relatively clear)
        tail = self.snake.body[-1]
        manhattan_to_tail = abs(head.x - tail.x) + abs(head.y - tail.y)
        # Normalize by snake length - if close relative to length, it's accessible
        tail_accessible = 1.0 if manhattan_to_tail <= len(self.snake.body) * SIZE * 0.5 else 0.0
        
        # Combine all state components
        state = (
            danger_8_dirs +              # [0-7]
            body_density_8_dirs +        # [8-15]
            current_direction +          # [16-19]
            apple_location +             # [20-23]
            [normalized_length,          # [24]
             body_density_immediate,     # [25]
             min_wall_distance,          # [26]
             tail_accessible]            # [27]
        )
        
        return np.array(state, dtype=np.float32)

    def _is_looping(self):
        """Detect if snake is going in circles."""
        if len(self.recent_positions) < 8:
            return False
        
        # Check if we've been to the same position recently
        head_pos = (self.snake.body[0].x, self.snake.body[0].y)
        return self.recent_positions.count(head_pos) > 2

    def _update_game_state(self, action=None):
        """
        Update game logic with enhanced reward shaping.
        """
        self.snake.can_change_direction = True
        prev_distance = self.prev_distance
        
        # Execute action if in AI mode
        if self.is_ai_mode and action is not None:
            move_map = {
                0: self.snake.move_up,
                1: self.snake.move_down,
                2: self.snake.move_left,
                3: self.snake.move_right
            }
            move_map[action]()
        
        self.snake.move()
        self.frame_iteration += 1
        
        # Track position for loop detection
        head_pos = (self.snake.body[0].x, self.snake.body[0].y)
        self.recent_positions.append(head_pos)
        if len(self.recent_positions) > 10:
            self.recent_positions.pop(0)

        game_over = False
        reward = 0
        
        # Check death
        if self.snake.die():
            game_over = True
            self.game_active = False
            reward = REWARD_DEATH
            return reward, game_over
        
        # Check apple eaten
        if self._check_colision():
            self.score += 1
            self.snake.add = True
            self.apple.generate(self.snake)
            reward = REWARD_EAT
            self.prev_distance = self._get_distance_to_apple()
            self.recent_positions = []  # Reset loop detection
        else:
            # Reward shaping: encourage moving towards apple
            current_distance = self._get_distance_to_apple()
            if current_distance < prev_distance:
                reward = REWARD_MOVE_CLOSER
            else:
                reward = REWARD_MOVE_AWAY
            self.prev_distance = current_distance
            
            # Penalty for looping
            if self._is_looping():
                reward += REWARD_LOOP
        
        # Timeout penalty (scale with snake length)
        max_moves = 100 * len(self.snake.body)
        if self.frame_iteration > max_moves:
            game_over = True
            self.game_active = False
            reward = REWARD_DEATH
            
        return reward, game_over

    def _game_over_screen(self):
        """Display game over screen."""
        title_text = self.TITLE_FONT.render("GAME OVER", 1, WHITE)
        score_text = self.SCORE_TEXT.render(
            f"Final Score: {self.score} | Length: {len(self.snake.body)}", 
            1, WHITE
        )
        restart_text = self.MENU_FONT.render("Press R to Restart (Human Mode)", 1, GREEN)
        ai_text = self.MENU_FONT.render("Press T to Restart (AI Training Mode)", 1, (100, 255, 100))
        quit_text = self.MENU_FONT.render("Press ESC or Q to Quit", 1, (255, 100, 100))

        self.WIN.fill(BLACK)

        self.WIN.blit(title_text, (ANCHO//2 - title_text.get_width()//2, ALTO//2 - 50))
        self.WIN.blit(score_text, (ANCHO//2 - score_text.get_width()//2, ALTO//2 - 10))
        self.WIN.blit(restart_text, (ANCHO//2 - restart_text.get_width()//2, ALTO//2 + 30))
        self.WIN.blit(ai_text, (ANCHO//2 - ai_text.get_width()//2, ALTO//2 + 50))
        self.WIN.blit(quit_text, (ANCHO//2 - quit_text.get_width()//2, ALTO//2 + 70))

        pygame.display.update()

        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._quit_game()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.is_ai_mode = False
                        return "RESTART"
                    if event.key == pygame.K_t:
                        self.is_ai_mode = True
                        return "RESTART"
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                        if self.is_ai_mode:
                            self.agent.save_model()
                        self._quit_game()
        return "QUIT"

    def _quit_game(self):
        """Quit game properly."""
        if self.is_ai_mode:
            self.agent.save_model()
        pygame.quit()
        quit()

    def run(self):
        """Main game loop with continuous AI training."""
        running = True
        
        print("\n" + "="*60)
        print("🐍 SNAKE AI V2 - Enhanced Spatial Awareness Training")
        print("="*60)
        print(f"📊 State Size: {STATE_SIZE} features")
        print(f"🎮 Actions: {ACTION_SIZE} (Up, Down, Left, Right)")
        print(f"🧠 Model: Deep Q-Network with spatial body tracking")
        print(f"💾 Saves: Every {self.save_frequency} episodes")
        print("="*60 + "\n")
        
        while running:
            self._init_elements()
            self.episode_count += 1
            
            # Active game loop
            while self.game_active:
                # Check for quit events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        if self.is_ai_mode:
                            self.agent.save_model()
                        self._quit_game()
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        if self.is_ai_mode:
                            self.agent.save_model()
                        self._quit_game()
                
                # Adjust FPS based on training progress
                if self.is_ai_mode:
                    if self.agent.epsilon < 0.1:
                        self.fps_clock.tick(FPS * 30)  # Very fast when trained
                    elif self.agent.epsilon < 0.5:
                        self.fps_clock.tick(FPS * 10)  # Fast
                    else:
                        self.fps_clock.tick(FPS * 5)   # Medium during early training
                else:
                    self.fps_clock.tick(FPS)

                if self.is_ai_mode:
                    # AI logic
                    old_state = self._get_state()
                    action = self.agent.act(old_state)
                    
                    reward, done = self._update_game_state(action)
                    new_state = self._get_state()
                    
                    self.agent.remember(old_state, action, reward, new_state, done)
                    self.agent.replay()
                    
                    if done:
                        self.total_score += self.score
                        if self.score > self.max_score:
                            self.max_score = self.score
                        
                        self.agent.on_episode_end()

                        avg_score = self.total_score / self.episode_count
                        print(f"Ep {self.episode_count:4d} | "
                              f"Score: {self.score:3d} | "
                              f"Len: {len(self.snake.body):3d} | "
                              f"Max: {self.max_score:3d} | "
                              f"Avg: {avg_score:5.1f} | "
                              f"ε: {self.agent.epsilon:.3f}")
                else:
                    # Human logic
                    self._handle_input()
                    self._update_game_state(action=None)
                
                self._draw_elements()

            # Save model periodically
            if self.is_ai_mode and self.episode_count % self.save_frequency == 0:
                self.agent.save_model()
                print(f"\n{'='*60}")
                print(f"📈 Checkpoint: {self.episode_count} episodes completed")
                print(f"🏆 Best Score: {self.max_score} | Avg: {self.total_score/self.episode_count:.1f}")
                print(f"{'='*60}\n")
                
            # In human mode, show game over screen
            if not self.is_ai_mode:
                action = self._game_over_screen()
                if action == "RESTART":
                    continue
                else:
                    running = False
            # In AI mode, automatically restart for continuous training

if __name__ == '__main__':
    game = Game(is_ai_mode=True)
    game.run()

