import torch
from train import train
import turtle
if __name__ == "__main__":
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    # Train model
    train(episodes=10000, render_every=100)  # render_every=100 means it will show progress every 100 episodes
    print("Training complete! Model saved as 'tetris_model.pth'")
