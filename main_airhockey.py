import pygame
import sys
from airhockey_game import AirHockeyGame
from train_airhockey import train_airhockey_dqn, test_vs_simple_ai
from dqn_agent import SimpleAIAgent

def play_vs_ai():
    """Play against simple AI"""
    game = AirHockeyGame()
    ai = SimpleAIAgent(difficulty=0.6)
    
    print("Controls:")
    print("WASD - Move paddle")
    print("R - Restart game")
    print("ESC - Exit")
    
    while game.running:
        game.handle_events()
        
        if game.state.value != 2:  # Not game over
            # Player controls (WASD)
            keys = pygame.key.get_pressed()
            p1_target = [game.paddle1.position.x, game.paddle1.position.y]
            
            if keys[pygame.K_w]:
                p1_target[1] -= game.paddle1.max_speed
            if keys[pygame.K_s]:
                p1_target[1] += game.paddle1.max_speed
            if keys[pygame.K_a]:
                p1_target[0] -= game.paddle1.max_speed
            if keys[pygame.K_d]:
                p1_target[0] += game.paddle1.max_speed
            
            # AI controls
            ai_target = ai.act(game, is_player1=False)
            
            game.update(tuple(p1_target), ai_target)
        
        game.render()
        game.clock.tick(60)
    
    pygame.quit()

def play_two_players():
    """Two player mode"""
    game = AirHockeyGame()
    
    print("Controls:")
    print("Player 1 (Blue): WASD")
    print("Player 2 (Red): Arrow Keys")
    print("R - Restart game")
    print("ESC - Exit")
    
    while game.running:
        game.handle_events()
        
        if game.state.value != 2:  # Not game over
            keys = pygame.key.get_pressed()
            
            # Player 1 (WASD)
            p1_target = [game.paddle1.position.x, game.paddle1.position.y]
            if keys[pygame.K_w]:
                p1_target[1] -= game.paddle1.max_speed
            if keys[pygame.K_s]:
                p1_target[1] += game.paddle1.max_speed
            if keys[pygame.K_a]:
                p1_target[0] -= game.paddle1.max_speed
            if keys[pygame.K_d]:
                p1_target[0] += game.paddle1.max_speed
            
            # Player 2 (Arrow keys)
            p2_target = [game.paddle2.position.x, game.paddle2.position.y]
            if keys[pygame.K_UP]:
                p2_target[1] -= game.paddle2.max_speed
            if keys[pygame.K_DOWN]:
                p2_target[1] += game.paddle2.max_speed
            if keys[pygame.K_LEFT]:
                p2_target[0] -= game.paddle2.max_speed
            if keys[pygame.K_RIGHT]:
                p2_target[0] += game.paddle2.max_speed
            
            game.update(tuple(p1_target), tuple(p2_target))
        
        game.render()
        game.clock.tick(60)
    
    pygame.quit()

def main():
    """Main menu"""
    print("=== DQN Air Hockey ===")
    print("1. Play vs Simple AI")
    print("2. Two Player Mode")
    print("3. Train DQN Agents")
    print("4. Test Trained Agent vs Simple AI")
    print("5. Watch Two AI Agents Play")
    print("6. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == "1":
                print("Starting game vs AI...")
                play_vs_ai()
                
            elif choice == "2":
                print("Starting two player game...")
                play_two_players()
                
            elif choice == "3":
                episodes = input("Enter number of episodes (default 1000): ").strip()
                episodes = int(episodes) if episodes.isdigit() else 1000
                
                print(f"Training DQN agents for {episodes} episodes...")
                print("This may take several hours depending on your hardware.")
                confirm = input("Continue? (y/n): ").strip().lower()
                
                if confirm == 'y':
                    train_airhockey_dqn(episodes=episodes)
                
            elif choice == "4":
                agent_path = input("Agent model path (default: airhockey_p1_final.pth): ").strip()
                agent_path = agent_path if agent_path else "airhockey_p1_final.pth"
                
                episodes = input("Number of test episodes (default 5): ").strip()
                episodes = int(episodes) if episodes.isdigit() else 5
                
                is_p1 = input("Is agent player 1? (y/n, default y): ").strip().lower()
                is_p1 = is_p1 != 'n'
                
                print("Testing trained agent vs Simple AI...")
                test_vs_simple_ai(agent_path, episodes, is_p1)
                
            elif choice == "5":
                print("Watching two AI agents play...")
                game = AirHockeyGame()
                ai1 = SimpleAIAgent(difficulty=0.8)
                ai2 = SimpleAIAgent(difficulty=0.7)
                
                print("Press ESC to exit")
                
                while game.running:
                    game.handle_events()
                    
                    if game.state.value != 2:  # Not game over
                        ai1_target = ai1.act(game, is_player1=True)
                        ai2_target = ai2.act(game, is_player1=False)
                        game.update(ai1_target, ai2_target)
                    
                    game.render()
                    game.clock.tick(60)
                
                pygame.quit()
                
            elif choice == "6":
                print("Goodbye!")
                break
                
            else:
                print("Invalid choice. Please enter 1-6.")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")

if __name__ == "__main__":
    main()