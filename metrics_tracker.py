#metrics_tracker.py
import json
import os
from datetime import datetime
from collections import defaultdict
from copy import deepcopy

METRICS_DIR = 'snake_models_v2'
METRICS_FILE = 'training_metrics.json'
METRICS_PATH = os.path.join(METRICS_DIR, METRICS_FILE)

class MetricsTracker:
    """
    Tracks detailed performance metrics for Snake AI training.
    """
    
    def __init__(self):
        self.current_session = {
            'start_time': datetime.now().isoformat(),
            'episodes': 0,
            'deaths': {
                'wall_collision': 0,        # Chocó con el borde
                'self_collision': 0,        # Se mordió a sí mismo
                'timeout': 0,               # Tardó demasiado (encerrado/loop)
                'total': 0
            },
            'scores': {
                'total': 0,
                'max': 0,
                'min': float('inf'),
                'last_100_avg': []
            },
            'snake_lengths': {
                'max': 3,
                'avg_at_death': [],
                'distribution': defaultdict(int)  # length: count
            },
            'survival_stats': {
                'total_steps': 0,
                'max_steps': 0,
                'avg_steps_per_episode': []
            },
            'apple_stats': {
                'total_apples': 0,
                'avg_steps_to_apple': [],
                'fastest_apple': float('inf'),
                'slowest_apple': 0
            },
            'spatial_issues': {
                'died_near_walls': 0,
                'died_in_corner': 0,
                'died_surrounded': 0
            },
            'efficiency': {
                'moves_per_apple': [],
                'apple_rate': []
            }
        }
        
        self.episode_data = {
            'score': 0,
            'steps': 0,
            'apples_eaten': 0,
            'steps_since_last_apple': 0
        }
        
        self.all_sessions = self._load_metrics()
    
    def _load_metrics(self):
        """Load existing metrics from file."""
        if os.path.exists(METRICS_PATH):
            try:
                with open(METRICS_PATH, 'r') as f:
                    data = json.load(f)
                    print(f"📊 Loaded existing metrics: {len(data.get('sessions', []))} previous sessions")
                    return data
            except Exception as e:
                print(f"⚠️ Error loading metrics: {e}")
        
        return {'sessions': []}
    
    def start_episode(self):
        """Reset episode-specific metrics."""
        self.episode_data = {
            'score': 0,
            'steps': 0,
            'apples_eaten': 0,
            'steps_since_last_apple': 0
        }
    
    def record_step(self):
        """Record a game step."""
        self.episode_data['steps'] += 1
        self.episode_data['steps_since_last_apple'] += 1
        self.current_session['survival_stats']['total_steps'] += 1
    
    def record_apple(self):
        """Record when an apple is eaten."""
        self.episode_data['score'] += 1
        self.episode_data['apples_eaten'] += 1
        
        steps_to_apple = self.episode_data['steps_since_last_apple']
        self.current_session['apple_stats']['total_apples'] += 1
        self.current_session['apple_stats']['avg_steps_to_apple'].append(steps_to_apple)
        
        if steps_to_apple < self.current_session['apple_stats']['fastest_apple']:
            self.current_session['apple_stats']['fastest_apple'] = steps_to_apple
        if steps_to_apple > self.current_session['apple_stats']['slowest_apple']:
            self.current_session['apple_stats']['slowest_apple'] = steps_to_apple
        
        self.episode_data['steps_since_last_apple'] = 0
    
    def record_death(self, death_type, snake_body, board_width, board_height, size):
        """
        Record death and analyze circumstances.
        """
        self.current_session['episodes'] += 1
        self.current_session['deaths']['total'] += 1
        
        # Record death type
        if death_type == 'wall':
            self.current_session['deaths']['wall_collision'] += 1
        elif death_type == 'self':
            self.current_session['deaths']['self_collision'] += 1
        elif death_type == 'timeout':
            self.current_session['deaths']['timeout'] += 1
        
        # Score statistics
        score = self.episode_data['score']
        self.current_session['scores']['total'] += score
        self.current_session['scores']['max'] = max(self.current_session['scores']['max'], score)
        self.current_session['scores']['min'] = min(self.current_session['scores']['min'], score)
        self.current_session['scores']['last_100_avg'].append(score)
        if len(self.current_session['scores']['last_100_avg']) > 100:
            self.current_session['scores']['last_100_avg'].pop(0)
        
        # Snake length statistics
        snake_length = len(snake_body)
        self.current_session['snake_lengths']['max'] = max(self.current_session['snake_lengths']['max'], snake_length)
        self.current_session['snake_lengths']['avg_at_death'].append(snake_length)
        self.current_session['snake_lengths']['distribution'][snake_length] += 1
        
        # Survival statistics
        steps = self.episode_data['steps']
        self.current_session['survival_stats']['max_steps'] = max(
            self.current_session['survival_stats']['max_steps'], steps)
        self.current_session['survival_stats']['avg_steps_per_episode'].append(steps)
        
        # Efficiency metrics
        if self.episode_data['apples_eaten'] > 0:
            moves_per_apple = steps / self.episode_data['apples_eaten']
            self.current_session['efficiency']['moves_per_apple'].append(moves_per_apple)
            
            apple_rate = (self.episode_data['apples_eaten'] / steps) * 100
            self.current_session['efficiency']['apple_rate'].append(apple_rate)
        
        # Spatial analysis
        head = snake_body[0]
        margin = size * 3
        near_wall = (
            head.x < margin or 
            head.x >= board_width - margin or 
            head.y < margin or 
            head.y >= board_height - margin
        )
        if near_wall:
            self.current_session['spatial_issues']['died_near_walls'] += 1
        
        corner_margin = size * 5
        in_corner = (
            (head.x < corner_margin and head.y < corner_margin) or
            (head.x < corner_margin and head.y >= board_height - corner_margin) or
            (head.x >= board_width - corner_margin and head.y < corner_margin) or
            (head.x >= board_width - corner_margin and head.y >= board_height - corner_margin)
        )
        if in_corner:
            self.current_session['spatial_issues']['died_in_corner'] += 1
        
        # Surrounded by body (3+ adjacent body parts)
        if len(snake_body) > 3:
            adjacent_body = 0
            for dx, dy in [(-size, 0), (size, 0), (0, -size), (0, size)]:
                check_pos = (head.x + dx, head.y + dy)
                for block in snake_body[1:]:
                    if (block.x, block.y) == check_pos:
                        adjacent_body += 1
                        break
            if adjacent_body >= 3:
                self.current_session['spatial_issues']['died_surrounded'] += 1
    
    def get_summary(self):
        """Get a formatted summary of current metrics."""
        episodes = self.current_session['episodes']
        if episodes == 0:
            return "No episodes completed yet."
        
        deaths = self.current_session['deaths']
        scores = self.current_session['scores']
        survival = self.current_session['survival_stats']
        spatial = self.current_session['spatial_issues']
        
        avg_score = scores['total'] / episodes
        avg_steps = sum(survival['avg_steps_per_episode']) / len(survival['avg_steps_per_episode'])
        wall_pct = (deaths['wall_collision'] / episodes) * 100
        self_pct = (deaths['self_collision'] / episodes) * 100
        timeout_pct = (deaths['timeout'] / episodes) * 100

        # ✅ Calcular valores condicionales antes de formatear
        last_100_avg = sum(scores['last_100_avg']) / len(scores['last_100_avg']) if scores['last_100_avg'] else 0
        avg_steps_to_apple = (
            sum(self.current_session['apple_stats']['avg_steps_to_apple']) /
            len(self.current_session['apple_stats']['avg_steps_to_apple'])
            if self.current_session['apple_stats']['avg_steps_to_apple'] else 0
        )
        avg_moves_per_apple = (
            sum(self.current_session['efficiency']['moves_per_apple']) /
            len(self.current_session['efficiency']['moves_per_apple'])
            if self.current_session['efficiency']['moves_per_apple'] else 0
        )
        avg_apple_rate = (
            sum(self.current_session['efficiency']['apple_rate']) /
            len(self.current_session['efficiency']['apple_rate'])
            if self.current_session['efficiency']['apple_rate'] else 0
        )
        
        summary = f"""
╔═══════════════════════════════════════════════════════════════╗
║           SNAKE AI PERFORMANCE METRICS SUMMARY                ║
╚═══════════════════════════════════════════════════════════════╝

📊 GENERAL STATS:
   Episodes Completed: {episodes}
   Total Steps: {survival['total_steps']:,}
   
🎯 SCORES:
   Average: {avg_score:.2f}
   Max: {scores['max']}
   Min: {scores['min']}
   Last 100 Avg: {last_100_avg:.2f}

💀 DEATH ANALYSIS:
   Wall Collisions: {deaths['wall_collision']} ({wall_pct:.1f}%)
   Self Collisions: {deaths['self_collision']} ({self_pct:.1f}%)
   Timeouts: {deaths['timeout']} ({timeout_pct:.1f}%)

🐍 SNAKE LENGTH:
   Max Length: {self.current_session['snake_lengths']['max']}
   Avg at Death: {sum(self.current_session['snake_lengths']['avg_at_death'])/len(self.current_session['snake_lengths']['avg_at_death']):.1f}

⏱️ SURVIVAL:
   Max Steps: {survival['max_steps']}
   Avg Steps/Episode: {avg_steps:.1f}

📍 SPATIAL ISSUES:
   Died Near Walls: {spatial['died_near_walls']} ({(spatial['died_near_walls']/episodes)*100:.1f}%)
   Died in Corners: {spatial['died_in_corner']} ({(spatial['died_in_corner']/episodes)*100:.1f}%)
   Died Surrounded: {spatial['died_surrounded']} ({(spatial['died_surrounded']/episodes)*100:.1f}%)

🍎 APPLE EFFICIENCY:
   Total Apples: {self.current_session['apple_stats']['total_apples']}
   Avg Steps to Apple: {avg_steps_to_apple:.1f}
   Fastest Apple: {self.current_session['apple_stats']['fastest_apple'] if self.current_session['apple_stats']['fastest_apple'] != float('inf') else 'N/A'}
   
⚡ EFFICIENCY:
   Avg Moves/Apple: {avg_moves_per_apple:.1f}
   Apple Rate: {avg_apple_rate:.2f} per 100 steps
"""
        return summary
    
    def save_metrics(self):
        """Save all metrics to JSON file."""
        # CAMBIO 2: Creamos una copia profunda de la sesión actual (session_to_save)
        # Esto asegura que el objeto original 'self.current_session' no se modifique.
        session_to_save = deepcopy(self.current_session) 
        session_to_save['end_time'] = datetime.now().isoformat()
        
        episodes = session_to_save['episodes']
        
        if episodes > 0:
            self.current_session['scores']['average'] = self.current_session['scores']['total'] / episodes
            
            if self.current_session['survival_stats']['avg_steps_per_episode']:
                self.current_session['survival_stats']['average'] = sum(
                    self.current_session['survival_stats']['avg_steps_per_episode']
                ) / len(self.current_session['survival_stats']['avg_steps_per_episode'])
        

        session_to_save['snake_lengths']['distribution'] = dict(
            session_to_save['snake_lengths']['distribution']
        )
        self.all_sessions['sessions'].append(session_to_save)
        
        try:
            os.makedirs(METRICS_DIR, exist_ok=True)
            with open(METRICS_PATH, 'w') as f:
                json.dump(self.all_sessions, f, indent=2)
            print(f"\n💾 Metrics saved to: {METRICS_PATH}")
        except Exception as e:
            print(f"\n❌ Error saving metrics: {e}")
    
    def print_summary(self):
        """Print formatted summary to console."""
        print(self.get_summary())


