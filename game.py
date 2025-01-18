from block import Block
import turtle
import numpy as np
SCALE = 32
class Game:
    def __init__(self, headless=True):
        self.headless = headless
        self.reset()
        
        if not headless:
            self.setup_display()
            turtle.clear()

    def reset(self):
        self.score = 0
        self.occupied = [[False for _ in range(10)] for _ in range(20)]
        self.active = Block()
        return self.get_state()

    def setup_display(self):
        turtle.setup(SCALE * 12 + 20, SCALE * 22 + 20)
        turtle.setworldcoordinates(-1.5, -1.5, 10.5, 20.5)
        turtle.hideturtle()
        turtle.delay(0)
        turtle.speed(0)
        turtle.tracer(0, 0)
        self.draw_play_area()

    def draw_play_area(self):
        turtle.bgcolor("black")
        turtle.pencolor("white")
        turtle.penup()
        turtle.setpos(-0.525, -0.525)
        turtle.pendown()
        for _ in range(2):
            turtle.forward(10.05)
            turtle.left(90)
            turtle.forward(20.05)
            turtle.left(90)
        turtle.update()

    def step(self, action):
        """
        Execute one step in the game.
        Actions: 0: Left, 1: Right, 2: Rotate, 3: Drop
        Returns: (new_state, reward, done)
        """
        reward = 0
        
        # Execute action
        if action == 0:   # Move left
            if self.active.valid(-1, 0, self.occupied):
                self.active.move(-1, 0)
        elif action == 1: # Move right
            if self.active.valid(1, 0, self.occupied):
                self.active.move(1, 0)
        elif action == 2: # Rotate
            self.rotate()
        elif action == 3: # Drop
            reward += self.drop()

        # Move block down and check game state
        game_over = not self.move_down()
        if game_over:
            reward -= 50  # Penalty for game over
        
        # Update display if not in headless mode
        if not self.headless:
            turtle.clear()  # Add this line
            self.draw_play_area()  # Add this line
            turtle.update()

        return self.get_state(), reward, game_over

    def move_down(self):
        """Move active block down one step. Returns False if game is over."""
        if self.active.valid(0, -1, self.occupied):
            self.active.move(0, -1)
            return True
            
        # Place block and check for game over
        for square in self.active.squares:
            x, y = int(square.xcor()), int(square.ycor())
            if y >= 19:
                return False
            self.occupied[y][x] = square
            
        # Clear lines and create new block
        lines_cleared = self.eliminate_lines()
        self.score += lines_cleared * 100
        self.active = Block()
        return True

    def rotate(self):
        if not self.active:
            return False
            
        center = self.active.squares[1]
        cx, cy = center.xcor(), center.ycor()
        
        new_positions = [(cx + (s.ycor() - cy), cy - (s.xcor() - cx)) 
                        for s in self.active.squares]
        
        if all(0 <= x <= 9 and y >= 0 and (y >= 20 or not self.occupied[int(y)][int(x)])
               for x, y in new_positions):
            for square, (x, y) in zip(self.active.squares, new_positions):
                square.goto(x, y)
            return True
        return False

    def drop(self):
        """Drop piece to bottom, return number of cells dropped"""
        drops = 0
        while self.active.valid(0, -1, self.occupied):
            self.active.move(0, -1)
            drops += 1
        return drops

    def eliminate_lines(self):
        """Eliminate completed lines and return number of lines cleared"""
        lines_cleared = 0
        y = 0
        while y < 20:
            if all(cell for cell in self.occupied[y]):
                lines_cleared += 1
                # Remove line
                for x in range(10):
                    if not self.headless:
                        self.occupied[y][x].hideturtle()
                # Move everything down
                for y2 in range(y + 1, 20):
                    for x in range(10):
                        self.occupied[y2-1][x] = self.occupied[y2][x]
                        if self.occupied[y2-1][x] and not self.headless:
                            self.occupied[y2-1][x].sety(y2-1)
                # Clear top line
                for x in range(10):
                    self.occupied[19][x] = False
            else:
                y += 1
        return lines_cleared

    def get_state(self):
        """Return current state for DQN input"""
        # Board state
        board = np.array([[1 if cell and not isinstance(cell, bool) else 0 
                          for cell in row] for row in self.occupied])
        
        # Active piece
        piece = np.zeros((20, 10))
        for square in self.active.squares:
            x, y = int(square.xcor()), int(square.ycor())
            if 0 <= y < 20:
                piece[y][x] = 1
                
        return np.stack([board, piece])
