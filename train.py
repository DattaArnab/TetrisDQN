import torch
import turtle
from game import Game
from dqn_agent import DQNAgent
def train(episodes=1000, render_every=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(device)
    best_score = 0
    screen = None  # Initialize screen variable
    for episode in range(episodes):
        try:
            if episode % render_every == 0:
                if screen:
                    screen.bye()
                turtle.TurtleScreen._RUNNING = True  # Reset terminator flag
                screen = turtle.Screen()  # Create new screen
                
            game = Game(headless=(episode % render_every != 0))
            state = game.reset()
            total_reward = 0
            steps = 0
            
            while True:
                action = agent.act(state)
                next_state, reward, done = game.step(action)
                agent.remember(state, action, reward, next_state, done)
                agent.replay()
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
                    
        except turtle.Terminator:
            continue
        
        if episode % 10 == 0:
            agent.update_target()
            
        if total_reward > best_score:
            best_score = total_reward
            torch.save(agent.model.state_dict(), 'best_tetris_model.pth')
            
        if episode % render_every == 0:
            print(f"Episode: {episode}, Score: {total_reward:.1f}, Steps: {steps}, Epsilon: {agent.epsilon:.3f}, Best: {best_score:.1f}")
            torch.save(agent.model.state_dict(), 'latest_tetris_model.pth')