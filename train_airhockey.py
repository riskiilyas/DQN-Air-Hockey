import pygame
import numpy as np
import matplotlib.pyplot as plt
from airhockey_game import AirHockeyGame, GameState
from dqn_agent import AirHockeyDQNAgent, SimpleAIAgent
import time
import os
import sys

class AirHockeyTrainingEnv:
    def __init__(self, render=False, headless=False):
        if headless:
            # Completely disable pygame for maximum speed
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
        
        self.render_mode = render and not headless
        self.headless = headless
        self.state_size = 15
        
        # Always create game, but control rendering
        self.game = AirHockeyGame()
        if headless:
            # Replace screen with dummy surface
            self.game.screen = pygame.Surface((100, 100))
    
    def reset(self):
        """Reset game and return initial states"""
        self.game.reset_game()
        p1_state = self.game.get_state_vector(for_player1=True)
        p2_state = self.game.get_state_vector(for_player1=False)
        return p1_state, p2_state
    
    def step(self, p1_action, p2_action, agent1, agent2):
        """Execute one step of the game"""
        # Get current positions
        p1_pos = (self.game.paddle1.position.x, self.game.paddle1.position.y)
        p2_pos = (self.game.paddle2.position.x, self.game.paddle2.position.y)
        
        # Convert actions to target positions
        p1_target = agent1.get_action_position(p1_action, p1_pos)
        p2_target = agent2.get_action_position(p2_action, p2_pos)
        
        # Store previous scores for reward calculation
        prev_score1 = self.game.score1
        prev_score2 = self.game.score2
        
        # Update game
        self.game.update(p1_target, p2_target)
        
        # Get new states
        p1_state = self.game.get_state_vector(for_player1=True)
        p2_state = self.game.get_state_vector(for_player1=False)
        
        # Calculate rewards
        p1_reward = self.game.get_reward(prev_score1, prev_score2, for_player1=True)
        p2_reward = self.game.get_reward(prev_score1, prev_score2, for_player1=False)
        
        # Check if done
        done = (self.game.state == GameState.GAME_OVER)
        
        return (p1_state, p2_state), (p1_reward, p2_reward), done
    
    def render(self):
        """Render the game"""
        if self.render_mode and not self.headless:
            self.game.render()

def handle_pygame_events():
    """Handle pygame events to prevent freezing"""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            print("\nTraining interrupted by user (window closed)")
            pygame.quit()
            sys.exit(0)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                print("\nTraining interrupted by ESC key")
                pygame.quit()
                sys.exit(0)
            elif event.key == pygame.K_SPACE:
                print("\nTraining paused. Press SPACE again to continue...")
                paused = True
                while paused:
                    for pause_event in pygame.event.get():
                        if pause_event.type == pygame.KEYDOWN and pause_event.key == pygame.K_SPACE:
                            paused = False
                            print("Training resumed.")
                        elif pause_event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit(0)
                    pygame.time.wait(50)

def train_visual_stable(episodes=1000, render_every=50, save_frequency=250):
    """Stable visual training with proper event handling"""
    
    print(f"Starting Visual Air Hockey DQN Training...")
    print(f"Episodes: {episodes}")
    print(f"Will render every {render_every} episodes")
    print(f"Controls: ESC=quit, SPACE=pause/resume")
    
    # Initialize pygame properly
    pygame.init()
    
    # Create agents
    agent1 = AirHockeyDQNAgent(
        state_size=15,
        action_size=25,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.9995,
        epsilon_min=0.1,
        batch_size=32
    )
    
    agent2 = AirHockeyDQNAgent(
        state_size=15,
        action_size=25,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.9995,
        epsilon_min=0.1,
        batch_size=32
    )
    
    # Training statistics
    episode_rewards_p1 = []
    episode_rewards_p2 = []
    episode_lengths = []
    wins_p1 = 0
    wins_p2 = 0
    draws = 0
    
    print(f"Device: {agent1.device}")
    start_time = time.time()
    last_print_time = start_time
    
    try:
        for episode in range(episodes):
            # Determine if this episode should be rendered
            should_render = (episode % render_every == 0)
            
            if should_render:
                print(f"\n--- Rendering Episode {episode} ---")
            
            # Create environment
            env = AirHockeyTrainingEnv(render=should_render, headless=not should_render)
            
            # Reset environment
            p1_state, p2_state = env.reset()
            
            episode_reward_p1 = 0
            episode_reward_p2 = 0
            steps = 0
            max_steps = 1200
            
            # Episode loop
            while steps < max_steps:
                # Handle pygame events to prevent freezing
                if should_render:
                    handle_pygame_events()
                
                # Get actions
                p1_action = agent1.act(p1_state)
                p2_action = agent2.act(p2_state)
                
                # Execute step
                next_states, rewards, done = env.step(p1_action, p2_action, agent1, agent2)
                next_p1_state, next_p2_state = next_states
                p1_reward, p2_reward = rewards
                
                # Store experiences
                agent1.remember(p1_state, p1_action, p1_reward, next_p1_state, done)
                agent2.remember(p2_state, p2_action, p2_reward, next_p2_state, done)
                
                # Update states
                p1_state = next_p1_state
                p2_state = next_p2_state
                
                # Accumulate rewards
                episode_reward_p1 += p1_reward
                episode_reward_p2 += p2_reward
                steps += 1
                
                # Train agents
                if steps % 4 == 0:
                    if len(agent1.memory) > agent1.batch_size:
                        agent1.replay()
                    if len(agent2.memory) > agent2.batch_size:
                        agent2.replay()
                
                # Render with controlled frequency
                if should_render:
                    if steps % 8 == 0:  # Render every 8 steps
                        env.render()
                        pygame.display.flip()
                        env.game.clock.tick(60)  # 60 FPS cap
                
                if done:
                    break
            
            # Episode finished
            if should_render:
                print(f"Episode {episode} finished: Score {env.game.score1}-{env.game.score2}, Steps: {steps}")
                # Show final state for a moment
                env.render()
                pygame.display.flip()
                time.sleep(1)
            
            # Record statistics
            episode_rewards_p1.append(episode_reward_p1)
            episode_rewards_p2.append(episode_reward_p2)
            episode_lengths.append(steps)
            
            # Count wins
            if env.game.score1 > env.game.score2:
                wins_p1 += 1
            elif env.game.score2 > env.game.score1:
                wins_p2 += 1
            else:
                draws += 1
            
            # Update target networks
            if episode % 100 == 0 and episode > 0:
                agent1.update_target_network()
                agent2.update_target_network()
            
            # Print progress
            current_time = time.time()
            if current_time - last_print_time >= 20:  # Print every 20 seconds
                episodes_per_sec = episode / (current_time - start_time)
                remaining_time = (episodes - episode) / max(episodes_per_sec, 0.001)
                
                avg_reward_p1 = np.mean(episode_rewards_p1[-50:]) if len(episode_rewards_p1) >= 50 else np.mean(episode_rewards_p1)
                avg_reward_p2 = np.mean(episode_rewards_p2[-50:]) if len(episode_rewards_p2) >= 50 else np.mean(episode_rewards_p2)
                avg_length = np.mean(episode_lengths[-50:]) if len(episode_lengths) >= 50 else np.mean(episode_lengths)
                
                print(f"\n=== Progress Report ===")
                print(f"Episode {episode}/{episodes} ({episode/episodes*100:.1f}%)")
                print(f"Speed: {episodes_per_sec:.2f} eps/sec, ETA: {remaining_time/60:.1f}m")
                print(f"P1: Reward={avg_reward_p1:.2f}, ε={agent1.epsilon:.3f}")
                print(f"P2: Reward={avg_reward_p2:.2f}, ε={agent2.epsilon:.3f}")
                print(f"Avg Length: {avg_length:.1f}")
                print(f"Wins: P1={wins_p1} ({wins_p1/(episode+1)*100:.1f}%), P2={wins_p2} ({wins_p2/(episode+1)*100:.1f}%), Draws={draws}")
                
                last_print_time = current_time
            
            # Save models
            if episode % save_frequency == 0 and episode > 0:
                agent1.save(f"airhockey_p1_episode_{episode}.pth")
                agent2.save(f"airhockey_p2_episode_{episode}.pth")
                print(f"Models saved at episode {episode}")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by Ctrl+C")
    except SystemExit:
        print("\nTraining stopped by user")
    finally:
        # Clean up
        pygame.quit()
        
        # Final save
        agent1.save("airhockey_p1_final.pth")
        agent2.save("airhockey_p2_final.pth")
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.1f} minutes")
        if episode > 0:
            print(f"Average speed: {episode/total_time:.2f} episodes/second")
        
        # Plot results
        if len(episode_rewards_p1) > 10:
            plot_training_results(episode_rewards_p1, episode_rewards_p2, episode_lengths, [], [])
    
    return agent1, agent2

def train_fast_headless(episodes=2000, save_frequency=500):
    """Ultra-fast headless training"""
    
    print(f"Starting Headless Training...")
    print(f"Episodes: {episodes}")
    
    # Disable pygame completely
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    pygame.init()
    
    # Create agents
    agent1 = AirHockeyDQNAgent(
        state_size=15,
        action_size=25,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.9995,
        epsilon_min=0.1,
        batch_size=32
    )
    
    agent2 = AirHockeyDQNAgent(
        state_size=15,
        action_size=25,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.9995,
        epsilon_min=0.1,
        batch_size=32
    )
    
    # Training statistics
    episode_rewards_p1 = []
    episode_rewards_p2 = []
    episode_lengths = []
    wins_p1 = 0
    wins_p2 = 0
    
    start_time = time.time()
    last_print_time = start_time
    
    try:
        for episode in range(episodes):
            # Create headless environment
            env = AirHockeyTrainingEnv(render=False, headless=True)
            
            # Reset environment
            p1_state, p2_state = env.reset()
            
            episode_reward_p1 = 0
            episode_reward_p2 = 0
            steps = 0
            max_steps = 1000
            
            # Episode loop
            while steps < max_steps:
                # Get actions
                p1_action = agent1.act(p1_state)
                p2_action = agent2.act(p2_state)
                
                # Execute step
                next_states, rewards, done = env.step(p1_action, p2_action, agent1, agent2)
                next_p1_state, next_p2_state = next_states
                p1_reward, p2_reward = rewards
                
                # Store experiences
                agent1.remember(p1_state, p1_action, p1_reward, next_p1_state, done)
                agent2.remember(p2_state, p2_action, p2_reward, next_p2_state, done)
                
                # Update states
                p1_state = next_p1_state
                p2_state = next_p2_state
                
                # Accumulate rewards
                episode_reward_p1 += p1_reward
                episode_reward_p2 += p2_reward
                steps += 1
                
                # Train agents
                if steps % 4 == 0:
                    if len(agent1.memory) > agent1.batch_size:
                        agent1.replay()
                    if len(agent2.memory) > agent2.batch_size:
                        agent2.replay()
                
                if done:
                    break
            
            # Record statistics
            episode_rewards_p1.append(episode_reward_p1)
            episode_rewards_p2.append(episode_reward_p2)
            episode_lengths.append(steps)
            
            # Count wins
            if env.game.score1 > env.game.score2:
                wins_p1 += 1
            elif env.game.score2 > env.game.score1:
                wins_p2 += 1
            
            # Update target networks
            if episode % 100 == 0 and episode > 0:
                agent1.update_target_network()
                agent2.update_target_network()
            
            # Print progress
            current_time = time.time()
            if current_time - last_print_time >= 30:  # Print every 30 seconds
                episodes_per_sec = episode / (current_time - start_time)
                remaining_time = (episodes - episode) / max(episodes_per_sec, 0.001)
                
                avg_reward_p1 = np.mean(episode_rewards_p1[-100:]) if len(episode_rewards_p1) >= 100 else np.mean(episode_rewards_p1)
                avg_reward_p2 = np.mean(episode_rewards_p2[-100:]) if len(episode_rewards_p2) >= 100 else np.mean(episode_rewards_p2)
                
                print(f"Episode {episode}/{episodes} ({episode/episodes*100:.1f}%) | "
                      f"Speed: {episodes_per_sec:.1f} eps/sec | "
                      f"ETA: {remaining_time/60:.1f}m | "
                      f"P1: {avg_reward_p1:.1f} | P2: {avg_reward_p2:.1f} | "
                      f"Wins: {wins_p1}/{wins_p2}")
                
                last_print_time = current_time
            
            # Save models
            if episode % save_frequency == 0 and episode > 0:
                agent1.save(f"airhockey_p1_episode_{episode}.pth")
                agent2.save(f"airhockey_p2_episode_{episode}.pth")
                print(f"Models saved at episode {episode}")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by Ctrl+C")
    finally:
        # Final save
        agent1.save("airhockey_p1_final.pth")
        agent2.save("airhockey_p2_final.pth")
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.1f} minutes")
        if episode > 0:
            print(f"Average speed: {episode/total_time:.2f} episodes/second")
    
    return agent1, agent2

def plot_training_results(rewards_p1, rewards_p2, episode_lengths, losses_p1, losses_p2):
    """Plot training statistics"""
    if len(rewards_p1) < 10:
        print("Not enough data to plot")
        return
        
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    window_size = min(50, len(rewards_p1) // 10)
    
    def smooth(data, window_size):
        if len(data) < window_size:
            return data
        smoothed = []
        for i in range(len(data)):
            start = max(0, i - window_size + 1)
            smoothed.append(np.mean(data[start:i+1]))
        return smoothed
    
    episodes = range(len(rewards_p1))
    
    # Rewards
    axes[0, 0].plot(episodes, smooth(rewards_p1, window_size), 'b-', label='Player 1', alpha=0.8)
    axes[0, 0].plot(episodes, smooth(rewards_p2, window_size), 'r-', label='Player 2', alpha=0.8)
    axes[0, 0].set_title('Training Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Episode lengths
    axes[0, 1].plot(episodes, smooth(episode_lengths, window_size), 'g-')
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].grid(True)
    
    # Reward difference
    reward_diff = [r1 - r2 for r1, r2 in zip(rewards_p1, rewards_p2)]
    axes[1, 0].plot(episodes, smooth(reward_diff, window_size), 'purple')
    axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].set_title('Reward Difference (P1 - P2)')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Difference')
    axes[1, 0].grid(True)
    
    # Win rate estimate
    win_estimate = []
    window = 100
    for i in range(len(rewards_p1)):
        start = max(0, i - window + 1)
        recent_p1 = rewards_p1[start:i+1]
        recent_p2 = rewards_p2[start:i+1]
        p1_better = sum(1 for r1, r2 in zip(recent_p1, recent_p2) if r1 > r2)
        win_rate = p1_better / len(recent_p1)
        win_estimate.append(win_rate)
    
    axes[1, 1].plot(episodes, win_estimate, 'orange')
    axes[1, 1].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('P1 Win Rate Estimate')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Win Rate')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('airhockey_training_results.png', dpi=200)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "visual":
            episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 500
            render_freq = int(sys.argv[3]) if len(sys.argv) > 3 else 25
            train_visual_stable(episodes=episodes, render_every=render_freq)
        elif mode == "fast":
            episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
            train_fast_headless(episodes=episodes)
        else:
            print("Unknown mode. Use 'visual' or 'fast'")
    else:
        print("Air Hockey DQN Training")
        print("Usage:")
        print("  python train_airhockey.py visual [episodes] [render_every] - Visual training")
        print("  python train_airhockey.py fast [episodes] - Fast headless training")
        print("\nExample:")
        print("  python train_airhockey.py visual 1000 50")
        print("  python train_airhockey.py fast 5000")