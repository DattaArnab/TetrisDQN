import torch
from game import Game
from dqn_agent import DQN


def play_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game = Game(headless=False)
    model = DQN().to(device)
    model.load_state_dict(torch.load('tetris_model.pth'))
    model.eval()
    
    state = game.reset()
    total_reward = 0
    
    while True:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = model(state_tensor).argmax().item()
        
        state, reward, done = game.step(action)
        total_reward += reward
        
        if done:
            break
    
    print(f"Game Over! Score: {total_reward}")