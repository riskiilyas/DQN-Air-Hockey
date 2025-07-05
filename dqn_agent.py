import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import math

class DQNNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQNNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class AirHockeyDQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, 
                 memory_size=50000, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        
        # Neural networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hidden_size = 512
        self.q_network = DQNNetwork(state_size, hidden_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, hidden_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Copy weights to target network
        self.update_target_network()
        
        # Action mapping for continuous control
        self.action_radius = 100  # Maximum movement radius
        self.create_action_space()
    
    def create_action_space(self):
        """Create discrete action space for continuous movement"""
        # Create grid of possible movements
        self.actions = []
        
        # Stay in place
        self.actions.append((0, 0))
        
        # 8 directional movements with different speeds
        directions = []
        for angle in range(0, 360, 45):  # 8 directions
            rad = math.radians(angle)
            for speed in [0.3, 0.6, 1.0]:  # 3 speeds
                dx = math.cos(rad) * speed * self.action_radius
                dy = math.sin(rad) * speed * self.action_radius
                directions.append((dx, dy))
        
        self.actions.extend(directions)
        self.action_size = len(self.actions)
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, current_position=None):
        """Choose action using epsilon-greedy policy"""
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Convert to numpy array if not already
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Set network to evaluation mode to handle single sample
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        self.q_network.train()
        
        return q_values.argmax().item()
    
    def get_action_position(self, action, current_position):
        """Convert action index to target position"""
        dx, dy = self.actions[action]
        target_x = current_position[0] + dx
        target_y = current_position[1] + dy
        return (target_x, target_y)
    
    def replay(self):
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return 0
        
        batch = random.sample(self.memory, self.batch_size)
        
        # Convert to numpy arrays first to avoid the warning
        states = np.array([e[0] for e in batch], dtype=np.float32)
        actions = np.array([e[1] for e in batch], dtype=np.int64)
        rewards = np.array([e[2] for e in batch], dtype=np.float32)
        next_states = np.array([e[3] for e in batch], dtype=np.float32)
        dones = np.array([e[4] for e in batch], dtype=bool)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def save(self, filename):
        """Save model weights"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'actions': self.actions
        }, filename)
    
    def load(self, filename):
        """Load model weights"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        if 'actions' in checkpoint:
            self.actions = checkpoint['actions']
            self.action_size = len(self.actions)

class SimpleAIAgent:
    """Simple AI agent for comparison"""
    def __init__(self, difficulty=0.8):
        self.difficulty = difficulty  # 0 to 1, how good the AI is
        self.reaction_time = max(1, int(10 * (1 - difficulty)))  # frames to react
        self.frame_counter = 0
        self.target_position = None
        self.prediction_frames = 5  # How many frames ahead to predict
    
    def predict_puck_position(self, puck, frames_ahead=5):
        """Predict where the puck will be in the future"""
        future_x = puck.position.x + puck.velocity.x * frames_ahead
        future_y = puck.position.y + puck.velocity.y * frames_ahead
        return (future_x, future_y)
    
    def act(self, game, is_player1=True):
        """Simple AI logic with improved prediction"""
        self.frame_counter += 1
        
        if is_player1:
            paddle = game.paddle1
            goal_y = game.paddle1.min_y + 50  # Defend goal area
            our_half_condition = lambda y: y < game.CENTER_LINE_Y
        else:
            paddle = game.paddle2
            goal_y = game.paddle2.max_y - 50  # Defend goal area
            our_half_condition = lambda y: y > game.CENTER_LINE_Y
        
        puck = game.puck
        
        # React every few frames based on difficulty
        if self.frame_counter % self.reaction_time == 0:
            # Predict puck position
            predicted_puck_pos = self.predict_puck_position(puck, self.prediction_frames)
            
            puck_in_our_half = our_half_condition(puck.position.y)
            
            if puck_in_our_half:
                # Aggressive: intercept the puck
                if abs(puck.velocity.x) > 1 or abs(puck.velocity.y) > 1:
                    # Puck is moving, try to intercept
                    self.target_position = predicted_puck_pos
                else:
                    # Puck is stationary, move directly to it
                    self.target_position = (puck.position.x, puck.position.y)
            else:
                # Defensive: stay near goal but track puck x-position
                target_x = predicted_puck_pos[0]
                # Add some defensive positioning
                if is_player1:
                    target_y = goal_y + random.uniform(-30, 30) * (1 - self.difficulty)
                else:
                    target_y = goal_y + random.uniform(-30, 30) * (1 - self.difficulty)
                
                self.target_position = (target_x, target_y)
            
            # Add some randomness based on difficulty
            if random.random() > self.difficulty:
                noise_x = random.uniform(-80, 80) * (1 - self.difficulty)
                noise_y = random.uniform(-40, 40) * (1 - self.difficulty)
                self.target_position = (
                    self.target_position[0] + noise_x,
                    self.target_position[1] + noise_y
                )
        
        return self.target_position if self.target_position else (paddle.position.x, paddle.position.y)