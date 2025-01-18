import turtle
SCALE = 32
class Square(turtle.Turtle):
    def __init__(self, x, y, color):
        super().__init__()
        self.shape("square")
        self.shapesize(SCALE / 20)
        self.speed(0)
        self.fillcolor(color)
        self.pencolor("gray")
        self.penup()
        self.goto(x, y)