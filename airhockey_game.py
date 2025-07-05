import pygame
import numpy as np
import math
import random
from typing import Tuple, List, Optional
from enum import Enum

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 600
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
DARK_GRAY = (64, 64, 64)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)

# Game Physics
FRICTION = 0.98
PADDLE_SPEED = 8
PUCK_MAX_SPEED = 15
BOUNCE_DAMPING = 0.85
PADDLE_FORCE = 12

# Game Objects
PADDLE_RADIUS = 25
PUCK_RADIUS = 12
GOAL_WIDTH = 120
GOAL_HEIGHT = 20

# Field dimensions
FIELD_MARGIN = 50
FIELD_WIDTH = WINDOW_WIDTH - 2 * FIELD_MARGIN
FIELD_HEIGHT = WINDOW_HEIGHT - 2 * FIELD_MARGIN
CENTER_LINE_Y = WINDOW_HEIGHT // 2

class GameState(Enum):
    PLAYING = 0
    GOAL_SCORED = 1
    GAME_OVER = 2

class Vector2D:
    def __init__(self, x: float = 0, y: float = 0):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        return Vector2D(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Vector2D(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        return Vector2D(self.x * scalar, self.y * scalar)
    
    def __truediv__(self, scalar):
        return Vector2D(self.x / scalar, self.y / scalar)
    
    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2)
    
    def normalize(self):
        mag = self.magnitude()
        if mag > 0:
            return Vector2D(self.x / mag, self.y / mag)
        return Vector2D(0, 0)
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y
    
    def distance_to(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def limit(self, max_magnitude):
        if self.magnitude() > max_magnitude:
            return self.normalize() * max_magnitude
        return Vector2D(self.x, self.y)
    
    def to_tuple(self):
        return (self.x, self.y)

class Paddle:
    def __init__(self, x: float, y: float, color: Tuple[int, int, int], is_player1: bool = True):
        self.position = Vector2D(x, y)
        self.velocity = Vector2D(0, 0)
        self.color = color
        self.radius = PADDLE_RADIUS
        self.is_player1 = is_player1
        self.max_speed = PADDLE_SPEED
        
        # Movement boundaries
        if is_player1:
            self.min_y = FIELD_MARGIN
            self.max_y = CENTER_LINE_Y - 10
        else:
            self.min_y = CENTER_LINE_Y + 10
            self.max_y = WINDOW_HEIGHT - FIELD_MARGIN
        
        self.min_x = FIELD_MARGIN + self.radius
        self.max_x = WINDOW_WIDTH - FIELD_MARGIN - self.radius
    
    def update(self, target_position: Optional[Vector2D] = None):
        """Update paddle position based on target or current velocity"""
        if target_position:
            # Move towards target position
            direction = target_position - self.position
            if direction.magnitude() > 1:
                self.velocity = direction.normalize() * self.max_speed
            else:
                self.velocity = Vector2D(0, 0)
        
        # Apply velocity
        self.position = self.position + self.velocity
        
        # Enforce boundaries
        self.position.x = max(self.min_x, min(self.max_x, self.position.x))
        self.position.y = max(self.min_y, min(self.max_y, self.position.y))
        
        # Apply friction
        self.velocity = self.velocity * FRICTION
    
    def move_towards(self, target_x: float, target_y: float):
        """Move paddle towards target position"""
        target = Vector2D(target_x, target_y)
        self.update(target)
    
    def draw(self, screen):
        """Draw paddle on screen"""
        pygame.draw.circle(screen, self.color, 
                         (int(self.position.x), int(self.position.y)), 
                         self.radius)
        pygame.draw.circle(screen, BLACK, 
                         (int(self.position.x), int(self.position.y)), 
                         self.radius, 2)

class Puck:
    def __init__(self, x: float, y: float):
        self.position = Vector2D(x, y)
        self.velocity = Vector2D(0, 0)
        self.radius = PUCK_RADIUS
        self.color = BLACK
        self.max_speed = PUCK_MAX_SPEED
    
    def update(self):
        """Update puck physics"""
        # Apply velocity
        self.position = self.position + self.velocity
        
        # Apply friction
        self.velocity = self.velocity * FRICTION
        
        # Limit speed
        self.velocity = self.velocity.limit(self.max_speed)
        
        # Bounce off walls (left and right)
        if (self.position.x - self.radius <= FIELD_MARGIN or 
            self.position.x + self.radius >= WINDOW_WIDTH - FIELD_MARGIN):
            self.velocity.x *= -BOUNCE_DAMPING
            self.position.x = max(FIELD_MARGIN + self.radius, 
                                min(WINDOW_WIDTH - FIELD_MARGIN - self.radius, 
                                    self.position.x))
        
        # Bounce off top and bottom walls (not goals)
        goal_left = WINDOW_WIDTH // 2 - GOAL_WIDTH // 2
        goal_right = WINDOW_WIDTH // 2 + GOAL_WIDTH // 2
        
        # Top wall
        if self.position.y - self.radius <= FIELD_MARGIN:
            if not (goal_left <= self.position.x <= goal_right):
                self.velocity.y *= -BOUNCE_DAMPING
                self.position.y = FIELD_MARGIN + self.radius
        
        # Bottom wall
        if self.position.y + self.radius >= WINDOW_HEIGHT - FIELD_MARGIN:
            if not (goal_left <= self.position.x <= goal_right):
                self.velocity.y *= -BOUNCE_DAMPING
                self.position.y = WINDOW_HEIGHT - FIELD_MARGIN - self.radius
    
    def check_goal(self) -> Optional[int]:
        """Check if puck scored a goal. Returns 1 for player1 goal, 2 for player2 goal"""
        goal_left = WINDOW_WIDTH // 2 - GOAL_WIDTH // 2
        goal_right = WINDOW_WIDTH // 2 + GOAL_WIDTH // 2
        
        # Player 1 goal (top)
        if (self.position.y - self.radius <= FIELD_MARGIN and 
            goal_left <= self.position.x <= goal_right):
            return 2  # Player 2 scored
        
        # Player 2 goal (bottom)
        if (self.position.y + self.radius >= WINDOW_HEIGHT - FIELD_MARGIN and 
            goal_left <= self.position.x <= goal_right):
            return 1  # Player 1 scored
        
        return None
    
    def reset_position(self):
        """Reset puck to center"""
        self.position = Vector2D(WINDOW_WIDTH // 2, (WINDOW_HEIGHT // 2) + random.choice([-50, 50]))
        self.velocity = Vector2D(0, 0)
    
    def draw(self, screen):
        """Draw puck on screen"""
        pygame.draw.circle(screen, self.color, 
                         (int(self.position.x), int(self.position.y)), 
                         self.radius)
        pygame.draw.circle(screen, WHITE, 
                         (int(self.position.x), int(self.position.y)), 
                         self.radius, 2)

class AirHockeyGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("DQN Air Hockey")
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Game objects
        self.paddle1 = Paddle(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 4, BLUE, True)
        self.paddle2 = Paddle(WINDOW_WIDTH // 2, 3 * WINDOW_HEIGHT // 4, RED, False)
        self.puck = Puck(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
        
        # Game state
        self.score1 = 0
        self.score2 = 0
        self.max_score = 7
        self.state = GameState.PLAYING
        self.goal_timer = 0
        self.goal_delay = 120  # frames
        
        # Font for UI
        self.font = pygame.font.Font(None, 36)
        self.big_font = pygame.font.Font(None, 72)
    
    def handle_collision(self, paddle: Paddle, puck: Puck):
        """Handle collision between paddle and puck"""
        distance = paddle.position.distance_to(puck.position)
        
        if distance < paddle.radius + puck.radius:
            # Calculate collision normal
            collision_normal = (puck.position - paddle.position).normalize()
            
            # Separate objects
            overlap = paddle.radius + puck.radius - distance
            puck.position = puck.position + collision_normal * overlap
            
            # Calculate relative velocity
            relative_velocity = puck.velocity - paddle.velocity
            
            # Calculate collision response
            impulse = relative_velocity.dot(collision_normal)
            if impulse > 0:  # Objects moving apart
                return
            
            # Apply impulse
            impulse_vector = collision_normal * impulse * 2
            puck.velocity = puck.velocity - impulse_vector
            
            # Add paddle momentum
            puck.velocity = puck.velocity + paddle.velocity * 0.5
            
            # Add some random variation
            random_factor = 0.2
            random_angle = random.uniform(-random_factor, random_factor)
            cos_a, sin_a = math.cos(random_angle), math.sin(random_angle)
            old_x, old_y = puck.velocity.x, puck.velocity.y
            puck.velocity.x = old_x * cos_a - old_y * sin_a
            puck.velocity.y = old_x * sin_a + old_y * cos_a
            
            # Limit puck speed
            puck.velocity = puck.velocity.limit(puck.max_speed)
    
    def update(self, paddle1_action: Optional[Tuple[float, float]] = None, 
               paddle2_action: Optional[Tuple[float, float]] = None):
        """Update game state"""
        if self.state == GameState.PLAYING:
            # Update paddles
            if paddle1_action:
                self.paddle1.move_towards(paddle1_action[0], paddle1_action[1])
            else:
                self.paddle1.update()
            
            if paddle2_action:
                self.paddle2.move_towards(paddle2_action[0], paddle2_action[1])
            else:
                self.paddle2.update()
            
            # Update puck
            self.puck.update()
            
            # Handle collisions
            self.handle_collision(self.paddle1, self.puck)
            self.handle_collision(self.paddle2, self.puck)
            
            # Check for goals
            goal = self.puck.check_goal()
            if goal:
                if goal == 1:
                    self.score1 += 1
                else:
                    self.score2 += 1
                
                self.state = GameState.GOAL_SCORED
                self.goal_timer = self.goal_delay
                
                # Check for game over
                if self.score1 >= self.max_score or self.score2 >= self.max_score:
                    self.state = GameState.GAME_OVER
        
        elif self.state == GameState.GOAL_SCORED:
            self.goal_timer -= 1
            if self.goal_timer <= 0:
                self.reset_round()
                self.state = GameState.PLAYING
    
    def reset_round(self):
        """Reset for next round"""
        self.puck.reset_position()
        self.paddle1.position = Vector2D(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 4)
        self.paddle2.position = Vector2D(WINDOW_WIDTH // 2, 3 * WINDOW_HEIGHT // 4)
        self.paddle1.velocity = Vector2D(0, 0)
        self.paddle2.velocity = Vector2D(0, 0)
    
    def reset_game(self):
        """Reset entire game"""
        self.score1 = 0
        self.score2 = 0
        self.state = GameState.PLAYING
        self.reset_round()
    
    def get_state_vector(self, for_player1: bool = True) -> np.ndarray:
        """Get state representation for DQN"""
        # Normalize positions to [-1, 1]
        puck_x = (self.puck.position.x - WINDOW_WIDTH/2) / (WINDOW_WIDTH/2)
        puck_y = (self.puck.position.y - WINDOW_HEIGHT/2) / (WINDOW_HEIGHT/2)
        puck_vx = self.puck.velocity.x / PUCK_MAX_SPEED
        puck_vy = self.puck.velocity.y / PUCK_MAX_SPEED
        
        paddle1_x = (self.paddle1.position.x - WINDOW_WIDTH/2) / (WINDOW_WIDTH/2)
        paddle1_y = (self.paddle1.position.y - WINDOW_HEIGHT/2) / (WINDOW_HEIGHT/2)
        paddle1_vx = self.paddle1.velocity.x / PADDLE_SPEED
        paddle1_vy = self.paddle1.velocity.y / PADDLE_SPEED
        
        paddle2_x = (self.paddle2.position.x - WINDOW_WIDTH/2) / (WINDOW_WIDTH/2)
        paddle2_y = (self.paddle2.position.y - WINDOW_HEIGHT/2) / (WINDOW_HEIGHT/2)
        paddle2_vx = self.paddle2.velocity.x / PADDLE_SPEED
        paddle2_vy = self.paddle2.velocity.y / PADDLE_SPEED
        
        # Distance from puck to each paddle
        dist_p1 = self.paddle1.position.distance_to(self.puck.position) / (WINDOW_WIDTH + WINDOW_HEIGHT)
        dist_p2 = self.paddle2.position.distance_to(self.puck.position) / (WINDOW_WIDTH + WINDOW_HEIGHT)
        
        # Score difference
        score_diff = (self.score1 - self.score2) / self.max_score
        
        state = np.array([
            puck_x, puck_y, puck_vx, puck_vy,
            paddle1_x, paddle1_y, paddle1_vx, paddle1_vy,
            paddle2_x, paddle2_y, paddle2_vx, paddle2_vy,
            dist_p1, dist_p2, score_diff
        ], dtype=np.float32)
        
        # Flip perspective for player 2
        if not for_player1:
            # Flip y coordinates and swap player positions
            state[1] = -state[1]  # puck_y
            state[3] = -state[3]  # puck_vy
            state[5] = -state[5]  # paddle1_y
            state[7] = -state[7]  # paddle1_vy
            state[9] = -state[9]  # paddle2_y
            state[11] = -state[11]  # paddle2_vy
            
            # Swap paddle data
            temp = state[4:8].copy()  # paddle1 data
            state[4:8] = state[8:12]  # paddle2 -> paddle1
            state[8:12] = temp        # paddle1 -> paddle2
            
            # Swap distances
            state[12], state[13] = state[13], state[12]
            
            # Flip score
            state[14] = -state[14]
        
        return state
    
    # Tambahkan method ini ke class AirHockeyGame
    def get_reward(self, previous_score1: int, previous_score2: int, for_player1: bool = True) -> float:
        """Calculate comprehensive reward for the given player"""
        reward = 0
        
        # Get paddle and opponent info
        if for_player1:
            paddle = self.paddle1
            opponent_paddle = self.paddle2
            our_score = self.score1
            opponent_score = self.score2
            score_change = self.score1 - previous_score1
            opponent_score_change = self.score2 - previous_score2
            our_goal_y = FIELD_MARGIN  # Top goal
            opponent_goal_y = WINDOW_HEIGHT - FIELD_MARGIN  # Bottom goal
        else:
            paddle = self.paddle2
            opponent_paddle = self.paddle1
            our_score = self.score2
            opponent_score = self.score1
            score_change = self.score2 - previous_score2
            opponent_score_change = self.score1 - previous_score1
            our_goal_y = WINDOW_HEIGHT - FIELD_MARGIN  # Bottom goal
            opponent_goal_y = FIELD_MARGIN  # Top goal
        
        # 1. SCORING REWARDS (Most Important)
        if score_change > 0:
            reward += 200  # Big reward for scoring
        if opponent_score_change > 0:
            reward -= 200  # Big penalty for opponent scoring
        
        # 2. PUCK INTERACTION REWARDS
        puck_paddle_distance = paddle.position.distance_to(self.puck.position)
        max_distance = 400  # Approximate max distance
        
        # Reward for being close to puck when it's moving slowly or in our territory
        puck_speed = self.puck.velocity.magnitude()
        
        if for_player1:
            puck_in_our_half = self.puck.position.y < CENTER_LINE_Y + 50  # Add buffer
            puck_in_opponent_half = self.puck.position.y > CENTER_LINE_Y - 50
        else:
            puck_in_our_half = self.puck.position.y > CENTER_LINE_Y - 50
            puck_in_opponent_half = self.puck.position.y < CENTER_LINE_Y + 50
        
        # Encourage approaching puck when it's in our territory or stationary
        if puck_in_our_half or puck_speed < 2:
            proximity_reward = max(0, 1 - puck_paddle_distance / max_distance) * 10
            reward += proximity_reward
        
        # 3. DEFENSIVE POSITIONING
        # Reward for staying between puck and our goal when puck is in opponent's half
        if puck_in_opponent_half:
            goal_center_x = WINDOW_WIDTH // 2
            paddle_goal_distance = abs(paddle.position.x - goal_center_x)
            ideal_defensive_y = our_goal_y + (80 if for_player1 else -80)
            paddle_defensive_distance = abs(paddle.position.y - ideal_defensive_y)
            
            defensive_reward = max(0, 1 - paddle_goal_distance / 200) * 1.5
            defensive_reward += max(0, 1 - paddle_defensive_distance / 100) * 1.5
            reward += defensive_reward
        
        # 4. OFFENSIVE POSITIONING
        # Reward for pushing puck toward opponent's goal
        if puck_in_our_half:
            # Check if we're pushing puck in right direction
            puck_to_goal_distance = abs(self.puck.position.y - opponent_goal_y)
            if for_player1:
                # We want puck moving down (positive y direction)
                if self.puck.velocity.y > 0:
                    reward += min(self.puck.velocity.y * 0.5, 3)
            else:
                # We want puck moving up (negative y direction)
                if self.puck.velocity.y < 0:
                    reward += min(abs(self.puck.velocity.y) * 0.5, 3)
        
        # 5. MOVEMENT REWARDS (Encourage active play)
        paddle_speed = paddle.velocity.magnitude()
        if paddle_speed > 0.5:
            reward += min(paddle_speed * 0.1, 0.5)  # Small reward for moving
        
        # 6. PENALTIES
        # Penalty for going to corners unnecessarily
        corner_penalty = 0
        margin = 80
        
        # Check if in corners
        in_corner = ((paddle.position.x < FIELD_MARGIN + margin or paddle.position.x > WINDOW_WIDTH - FIELD_MARGIN - margin) and
                    (paddle.position.y < FIELD_MARGIN + margin or paddle.position.y > WINDOW_HEIGHT - FIELD_MARGIN - margin))
        
        if in_corner:
            # Only penalize if puck is not nearby
            if puck_paddle_distance > 100:
                corner_penalty = -2
        
        reward += corner_penalty
        
        # Penalty for staying idle when puck is close
        if puck_paddle_distance < 80 and paddle_speed < 0.5:
            reward -= 1
        
        # 7. BOUNDARY RESPECT
        # Small penalty for being too close to boundaries when not necessary
        boundary_penalty = 0
        safe_margin = 40
        
        if (paddle.position.x < FIELD_MARGIN + safe_margin or 
            paddle.position.x > WINDOW_WIDTH - FIELD_MARGIN - safe_margin or
            paddle.position.y < paddle.min_y + safe_margin or 
            paddle.position.y > paddle.max_y - safe_margin):
            if puck_paddle_distance > 60:  # Only penalize if puck is not close
                boundary_penalty = -0.5
        
        reward += boundary_penalty
        
        # 8. GAME STATE REWARDS
        # Small ongoing reward for being ahead
        if our_score > opponent_score:
            reward += 0.2
        elif our_score < opponent_score:
            reward -= 0.1
        
        # Bonus for maintaining control (puck moving slowly in our favor)
        if puck_speed < 3 and puck_in_our_half:
            reward += 0.5
        
        return reward

    # Juga tambahkan method untuk debugging
    def get_reward_breakdown(self, previous_score1: int, previous_score2: int, for_player1: bool = True) -> dict:
        """Get detailed breakdown of reward components for analysis"""
        breakdown = {}
        
        if for_player1:
            paddle = self.paddle1
            score_change = self.score1 - previous_score1
            opponent_score_change = self.score2 - previous_score2
        else:
            paddle = self.paddle2
            score_change = self.score2 - previous_score2
            opponent_score_change = self.score1 - previous_score1
        
        # Calculate each component
        breakdown['scoring'] = 200 if score_change > 0 else 0
        breakdown['opponent_scoring'] = -200 if opponent_score_change > 0 else 0
        
        puck_distance = paddle.position.distance_to(self.puck.position)
        breakdown['proximity'] = max(0, 1 - puck_distance / 400) * 3
        
        paddle_speed = paddle.velocity.magnitude()
        breakdown['movement'] = min(paddle_speed * 0.1, 0.5) if paddle_speed > 0.5 else 0
        
        # Add other components...
        breakdown['total'] = sum(breakdown.values())
        
        return breakdown
    
    def draw_field(self):
        """Draw the air hockey field"""
        # Background
        self.screen.fill(WHITE)
        
        # Field boundaries
        field_rect = pygame.Rect(FIELD_MARGIN, FIELD_MARGIN, FIELD_WIDTH, FIELD_HEIGHT)
        pygame.draw.rect(self.screen, LIGHT_GRAY, field_rect)
        pygame.draw.rect(self.screen, BLACK, field_rect, 3)
        
        # Center line
        pygame.draw.line(self.screen, BLACK, 
                        (FIELD_MARGIN, CENTER_LINE_Y), 
                        (WINDOW_WIDTH - FIELD_MARGIN, CENTER_LINE_Y), 2)
        
        # Center circle
        pygame.draw.circle(self.screen, BLACK, 
                         (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2), 50, 2)
        
        # Goals
        goal_left = WINDOW_WIDTH // 2 - GOAL_WIDTH // 2
        goal_right = WINDOW_WIDTH // 2 + GOAL_WIDTH // 2
        
        # Top goal
        pygame.draw.rect(self.screen, RED, 
                        (goal_left, FIELD_MARGIN - GOAL_HEIGHT, GOAL_WIDTH, GOAL_HEIGHT))
        pygame.draw.rect(self.screen, BLACK, 
                        (goal_left, FIELD_MARGIN - GOAL_HEIGHT, GOAL_WIDTH, GOAL_HEIGHT), 2)
        
        # Bottom goal
        pygame.draw.rect(self.screen, BLUE, 
                        (goal_left, WINDOW_HEIGHT - FIELD_MARGIN, GOAL_WIDTH, GOAL_HEIGHT))
        pygame.draw.rect(self.screen, BLACK, 
                        (goal_left, WINDOW_HEIGHT - FIELD_MARGIN, GOAL_WIDTH, GOAL_HEIGHT), 2)
    
    def draw_ui(self):
        """Draw UI elements"""
        # Score
        score_text = self.big_font.render(f"{self.score1} - {self.score2}", True, BLACK)
        score_rect = score_text.get_rect(center=(WINDOW_WIDTH // 2, 30))
        self.screen.blit(score_text, score_rect)
        
        # Player labels
        p1_text = self.font.render("Player 1 (Blue)", True, BLUE)
        p1_rect = p1_text.get_rect(center=(WINDOW_WIDTH // 4, WINDOW_HEIGHT - 20))
        self.screen.blit(p1_text, p1_rect)
        
        p2_text = self.font.render("Player 2 (Red)", True, RED)
        p2_rect = p2_text.get_rect(center=(3 * WINDOW_WIDTH // 4, WINDOW_HEIGHT - 20))
        self.screen.blit(p2_text, p2_rect)
        
        # Game over screen
        if self.state == GameState.GAME_OVER:
            overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
            overlay.set_alpha(128)
            overlay.fill(BLACK)
            self.screen.blit(overlay, (0, 0))
            
            winner = "Player 1" if self.score1 > self.score2 else "Player 2"
            winner_text = self.big_font.render(f"{winner} Wins!", True, WHITE)
            winner_rect = winner_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
            self.screen.blit(winner_text, winner_rect)
            
            restart_text = self.font.render("Press R to restart", True, WHITE)
            restart_rect = restart_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 60))
            self.screen.blit(restart_text, restart_rect)
    
    def render(self):
        """Render the game"""
        self.draw_field()
        
        # Draw game objects
        self.paddle1.draw(self.screen)
        self.paddle2.draw(self.screen)
        self.puck.draw(self.screen)
        
        self.draw_ui()
        pygame.display.flip()
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.reset_game()
                elif event.key == pygame.K_ESCAPE:
                    self.running = False
    
    def run_manual(self):
        """Run game with manual controls"""
        while self.running:
            self.handle_events()
            
            # Manual controls
            keys = pygame.key.get_pressed()
            
            # Player 1 (WASD)
            p1_target = Vector2D(self.paddle1.position.x, self.paddle1.position.y)
            if keys[pygame.K_w]:
                p1_target.y -= PADDLE_SPEED
            if keys[pygame.K_s]:
                p1_target.y += PADDLE_SPEED
            if keys[pygame.K_a]:
                p1_target.x -= PADDLE_SPEED
            if keys[pygame.K_d]:
                p1_target.x += PADDLE_SPEED
            
            # Player 2 (Arrow keys)
            p2_target = Vector2D(self.paddle2.position.x, self.paddle2.position.y)
            if keys[pygame.K_UP]:
                p2_target.y -= PADDLE_SPEED
            if keys[pygame.K_DOWN]:
                p2_target.y += PADDLE_SPEED
            if keys[pygame.K_LEFT]:
                p2_target.x -= PADDLE_SPEED
            if keys[pygame.K_RIGHT]:
                p2_target.x += PADDLE_SPEED
            
            self.update((p1_target.x, p1_target.y), (p2_target.x, p2_target.y))
            self.render()
            self.clock.tick(FPS)
        
        pygame.quit()

if __name__ == "__main__":
    game = AirHockeyGame()
    game.run_manual()