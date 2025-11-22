import pygame
from pygame.math import Vector2
from collections import deque
import numpy as np

from snake import Snake
from apple import Apple
from dqn_agent import DQNAgent
from metrics_tracker import MetricsTracker

from constants import ANCHO, ALTO, SIZE, RED, GREEN, BLACK, WHITE, FPS

REWARD_EAT = 10
REWARD_MOVE_CLOSER = 1.0
REWARD_MOVE_AWAY = -0.5
REWARD_DEATH = -10
REWARD_LOOP = -5
REWARD_DANGER_ZONE = -2.0
REWARD_ESCAPE_ROUTE = 0.3
REWARD_NO_PATH = -8
# ============================================================================

STATE_SIZE = 28
ACTION_SIZE = 4

pygame.init()

class Game:
    def __init__(self, is_ai_mode=True):
        self.is_ai_mode = is_ai_mode
        
        self.WIN = pygame.display.set_mode((ANCHO, ALTO))
        pygame.display.set_caption("Snake AI V2.1 - Improved Spatial Planning")

        self.SCORE_TEXT = pygame.font.SysFont("Russo One", 12)
        self.TITLE_FONT = pygame.font.SysFont("Russo One", 30)
        self.MENU_FONT = pygame.font.SysFont("Russo One", 15)

        self.fps_clock = pygame.time.Clock()

        if self.is_ai_mode:
            self.agent = DQNAgent(STATE_SIZE, ACTION_SIZE)
            self.metrics = MetricsTracker()
        
        self._init_elements()
        
        self.episode_count = 0
        self.total_score = 0
        self.max_score = 0
        self.save_frequency = 2
        
        self.recent_positions = []
        self.metrics_save_frequency = 10
        self._trapped_this_episode = False

    def _init_elements(self):
        """Initialize game elements."""
        self.snake = Snake()
        self.apple = Apple(self.snake)
        self.score = 0
        self.game_active = True
        self.frame_iteration = 0
        self.prev_distance = self._get_distance_to_apple()
        self.recent_positions = []
        self._trapped_this_episode = False

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
                f"Length: {len(self.snake.body)} | Epsilon: {self.agent.epsilon:.3f}",
                f"Steps: {self.frame_iteration} | Apples: {self.metrics.episode_data['apples_eaten']}"
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
        if point.x < 0 or point.x >= ANCHO or point.y < 0 or point.y >= ALTO:
            return 1
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
        
        return min(count / max(len(self.snake.body) - 1, 1), 1.0)

    def _get_state(self):

        head = self.snake.body[0]
        
        dir_l = self.snake.direction_left
        dir_r = self.snake.direction_right
        dir_u = self.snake.direction_up
        dir_d = self.snake.direction_down
        
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
            head.x / ANCHO,
            (ANCHO - head.x) / ANCHO,
            head.y / ALTO,
            (ALTO - head.y) / ALTO
        ]
        min_wall_distance = min(dist_to_walls)
        
        # [27] Can reach tail?
        tail = self.snake.body[-1]
        manhattan_to_tail = abs(head.x - tail.x) + abs(head.y - tail.y)
        tail_accessible = 1.0 if manhattan_to_tail <= len(self.snake.body) * SIZE * 0.5 else 0.0
        
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
        """
        🔥 MEJORADO: Detecta loops más temprano
        Reduce de 8 a 6 posiciones para detectar problemas antes
        """
        if len(self.recent_positions) < 6:  # Era 8
            return False
        
        head_pos = (self.snake.body[0].x, self.snake.body[0].y)
        return self.recent_positions.count(head_pos) > 2

    def _get_path_length_to_tail(self):
        """
        Calcula la longitud del camino más corto (BFS) desde la cabeza hasta la cola.
        Retorna la longitud del camino (en bloques) o infinito si no hay camino.
        """
        head = self.snake.body[0]
        target = self.snake.body[-1]  # La cola es el objetivo

        if head == target:
            return 0

        # Posiciones ocupadas por el cuerpo (excluyendo la cola)
        occupied = {(block.x, block.y) for block in self.snake.body[:-1]}

        # BFS setup: (x, y, distance)
        queue = deque([(head.x, head.y, 0)])
        visited = {(head.x, head.y)}

        directions = [
            (0, -SIZE),
            (0, SIZE),
            (-SIZE, 0),
            (SIZE, 0)
        ]

        # BFS algorithm
        while queue:
            current_x, current_y, dist = queue.popleft()

            for dx, dy in directions:
                next_x = current_x + dx
                next_y = current_y + dy
                next_pos_tuple = (next_x, next_y)

                # 1. Validar límites
                if next_x < 0 or next_x >= ANCHO or next_y < 0 or next_y >= ALTO:
                    continue

                # 2. Validar colisión con el cuerpo *ocupado* (no la cola)
                if next_pos_tuple in occupied:
                    continue

                # 3. Validar no visitado
                if next_pos_tuple in visited:
                    continue

                # 4. ¡Encontramos la cola!
                if next_x == target.x and next_y == target.y:
                    return dist + 1  # Distancia de la cabeza al objetivo (en pasos)

                # Marcar como visitado y agregar a la cola
                visited.add(next_pos_tuple)
                queue.append((next_x, next_y, dist + 1))

        # No se encontró ningún camino
        return float('inf')
    

    def _has_path_to_apple(self):
        head = self.snake.body[0]
        target = self.apple.pos
        
        if head == target:
            self._trapped_this_episode = False
            return True
        
        occupied = {(block.x, block.y) for block in self.snake.body[:-1]}
        
        # BFS setup
        # ### MODIFICADO: Agregamos un tercer valor '0' que representa los pasos dados (distancia)
        queue = deque([(head.x, head.y, 0)])
        visited = {(head.x, head.y)}
        
        directions = [
            (0, -SIZE), 
            (0, SIZE),  
            (-SIZE, 0),
            (SIZE, 0)
        ]
        
        while queue:
            # ### MODIFICADO: Desempaquetamos ahora 3 valores
            current_x, current_y, dist = queue.popleft()
            
            # ### NUEVO: Si ya hemos caminado más de 20 pasos, consideramos que hay salida
            if dist > 20:
                self._trapped_this_episode = False
                return True
            
            for dx, dy in directions:
                next_x = current_x + dx
                next_y = current_y + dy
                
                if next_x < 0 or next_x >= ANCHO or next_y < 0 or next_y >= ALTO:
                    continue
                
                if (next_x, next_y) in occupied:
                    continue
                
                if (next_x, next_y) in visited:
                    continue
                
                if next_x == target.x and next_y == target.y:
                    self._trapped_this_episode = False
                    return True
                
                visited.add((next_x, next_y))
                # ### MODIFICADO: Agregamos el vecino con la distancia + 1
                queue.append((next_x, next_y, dist + 1))
        
        print("Esta encerrada")
        return False

    def _determine_death_type(self):
        """
        Determine the type of death that occurred.
        Returns: 'wall', 'self', or 'timeout'
        """
        head = self.snake.body[0]
        
        if (head.x < 0 or head.x >= ANCHO or head.y < 0 or head.y >= ALTO):
            return 'wall'
        
        for block in self.snake.body[1:]:
            if head == block:
                return 'self'
        
        return 'timeout'

    def _update_game_state(self, action=None):

        self.snake.can_change_direction = True
        prev_distance = self.prev_distance
        
        if self.is_ai_mode and action is not None:
            move_map = {
                0: self.snake.move_up,
                1: self.snake.move_down,
                2: self.snake.move_left,
                3: self.snake.move_right
            }
            move_map[action]()
        
        next_point = Vector2(
            self.snake.body[0].x + self.snake.direction.x, 
            self.snake.body[0].y + self.snake.direction.y
        )
        
        self.snake.move()
        self.frame_iteration += 1
        
        if self.is_ai_mode:
            self.metrics.record_step()
        
        head_pos = (self.snake.body[0].x, self.snake.body[0].y)
        self.recent_positions.append(head_pos)
        if len(self.recent_positions) > 10:
            self.recent_positions.pop(0)

        game_over = False
        reward = 0
        
        # Check death PRIMERO (colisión física real)
        if self.snake.die():
            game_over = True
            self.game_active = False
            reward = REWARD_DEATH
            
            if self.is_ai_mode:
                death_type = self._determine_death_type()
                self.metrics.record_death(
                    death_type=death_type,
                    snake_body=self.snake.body,
                    board_width=ANCHO,
                    board_height=ALTO,
                    size=SIZE
                )
            
            return reward, game_over
        
        # Check apple eaten
        if self._check_colision():
            self.score += 1
            self.snake.add = True
            self.apple.generate(self.snake)
            reward = REWARD_EAT
            self.prev_distance = self._get_distance_to_apple()
            self.recent_positions = []
            
            if self.is_ai_mode:
                self.metrics.record_apple()
        else:
            # ============================================================
            # 🔥 SISTEMA DE REWARDS MEJORADO
            # ============================================================
            current_distance = self._get_distance_to_apple()
            
            # 1. Reward base por dirección hacia la manzana
            if current_distance < prev_distance:
                reward = REWARD_MOVE_CLOSER
            else:
                reward = REWARD_MOVE_AWAY

            if reward == REWARD_MOVE_AWAY:
                # Obtenemos la longitud real del camino más corto a la cola
                path_to_tail_len = self._get_path_length_to_tail()
                snake_len = len(self.snake.body)
                
                # Definimos un umbral: ¿Es un camino corto y útil? 
                # e.g., menos de la mitad de la longitud total, o un valor fijo (ej. 10)
                TAIL_PATH_THRESHOLD = max(5, snake_len // 2)
                
                if path_to_tail_len <= TAIL_PATH_THRESHOLD:
                    # El movimiento de "alejarse" se compensa porque asegura la ruta de escape.
                    reward += REWARD_ESCAPE_ROUTE  # (0.3)
            
            # ============================================================
            # 🔥 VALIDACIÓN DE CAMINO: Solo penaliza, NO termina el juego
            # ============================================================
            if not self._has_path_to_apple() and not self._trapped_this_episode:
                reward += REWARD_NO_PATH
                self._trapped_this_episode = True
            # ============================================================
            
            # 3. Penalización por entrar a zona peligrosa
            new_head = self.snake.body[0]
            body_count_around = 0
            for dx in [-SIZE, 0, SIZE]:
                for dy in [-SIZE, 0, SIZE]:
                    if dx == 0 and dy == 0:
                        continue
                    check_point = Vector2(new_head.x + dx, new_head.y + dy)
                    if check_point in self.snake.body[1:]:
                        body_count_around += 1
            
            body_density = body_count_around / 8
            
            if body_density > 0.5:
                reward += REWARD_DANGER_ZONE
            
            
            self.prev_distance = current_distance
            
            # 5. Penalización por loops
            if self._is_looping():
                reward += REWARD_LOOP
            # ============================================================
        
        # Timeout penalty
        max_moves = 100 * len(self.snake.body)
        if self.frame_iteration > max_moves:
            game_over = True
            self.game_active = False
            reward = REWARD_DEATH
            
            if self.is_ai_mode:
                self.metrics.record_death(
                    death_type='timeout',
                    snake_body=self.snake.body,
                    board_width=ANCHO,
                    board_height=ALTO,
                    size=SIZE
                )
            
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
                            self.metrics.save_metrics()
                        self._quit_game()
        return "QUIT"

    def _quit_game(self):
        """Quit game properly."""
        if self.is_ai_mode:
            self.agent.save_model()
            self.metrics.save_metrics()
            print("\n" + "="*65)
            self.metrics.print_summary()
            print("="*65)
        pygame.quit()
        quit()

    def run(self):
        """Main game loop with continuous AI training."""
        running = True
        
        print("\n" + "="*70)
        print("🐍 SNAKE AI V2.1 - IMPROVED SPATIAL PLANNING & REWARDS")
        print("="*70)
        print(f"📊 State Size: {STATE_SIZE} features")
        print(f"🎮 Actions: {ACTION_SIZE} (Up, Down, Left, Right)")
        print(f"🧠 Model: Deep Q-Network (loading existing model)")
        print(f"💾 Saves: Every {self.save_frequency} episodes")
        print(f"📈 Metrics: Tracking {self.metrics_save_frequency} episode intervals")
        print("="*70 + "\n")
        
        while running:
            self._init_elements()
            self.episode_count += 1
            
            if self.is_ai_mode:
                self.metrics.start_episode()
            
            while self.game_active:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        if self.is_ai_mode:
                            self.agent.save_model()
                            self.metrics.save_metrics()
                            self.metrics.print_summary()
                        self._quit_game()
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        if self.is_ai_mode:
                            self.agent.save_model()
                            self.metrics.save_metrics()
                            self.metrics.print_summary()
                        self._quit_game()
                
                if self.is_ai_mode:
                    if self.agent.epsilon < 0.1:
                        self.fps_clock.tick(FPS * 30)
                    elif self.agent.epsilon < 0.5:
                        self.fps_clock.tick(FPS * 10)
                    else:
                        self.fps_clock.tick(FPS * 5)
                else:
                    self.fps_clock.tick(FPS)

                if self.is_ai_mode:
                    old_state = self._get_state()
                    action = self.agent.act(old_state)
                    
                    reward, done = self._update_game_state(action)
                    new_state = self._get_state()
                    
                    # 🔥 ACTIVAR ESTAS LÍNEAS PARA ENTRENAR
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
                    self._handle_input()
                    self._update_game_state(action=None)
                
                self._draw_elements()

            if self.is_ai_mode and self.episode_count % self.save_frequency == 0:
                self.agent.save_model()
                print(f"\n{'='*60}")
                print(f"📈 Checkpoint: {self.episode_count} episodes completed")
                print(f"🏆 Best Score: {self.max_score} | Avg: {self.total_score/self.episode_count:.1f}")
                print(f"{'='*60}\n")
            
            if self.is_ai_mode and self.episode_count % self.metrics_save_frequency == 0:
                self.metrics.save_metrics()
                print("\n")
                self.metrics.print_summary()
                
            if not self.is_ai_mode:
                action = self._game_over_screen()
                if action == "RESTART":
                    continue
                else:
                    running = False

if __name__ == '__main__':
    game = Game(is_ai_mode=True)
    game.run()

