import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

# Constants
WIDTH, HEIGHT = 800, 600
BALL_RADIUS = 1
ENCL_X = (-10, 10)
ENCL_Y = (-8, 8)
ENCL_Z = (-6, 6)
INIT_POS = (0, 0, -2)
INIT_VELOCITY = (0.3, 0.5, 0.2)

class Ball:
    def __init__(self, position, velocity):
        self.position = np.array(position, dtype='float32')
        self.velocity = np.array(velocity, dtype='float32')

    def move(self):
        self.position += self.velocity

    def reflect_if_needed(self):
        for i in range(3):
            if self.position[i] < (-1 if i == 0 else (-1 if i == 1 else -6)) or self.position[i] > (1 if i == 0 else (8 if i == 1 else 6)):
                self.velocity[i] = -self.velocity[i]

def draw_ball(ball):
    quadric = gluNewQuadric()
    glColor3f(1, 0, 0)  # Red color for the ball
    glPushMatrix()
    glTranslatef(*ball.position)
    gluSphere(quadric, BALL_RADIUS, 32, 32)
    glPopMatrix()

def draw_enclosure():
    glColor3f(0, 0, 1)  # Blue color for the enclosure
    glBegin(GL_LINES)
    # Draw the edges of the cuboid
    edges = [
        ((-10, -8, -6), (10, -8, -6)),
        ((-10, 8, -6), (10, 8, -6)),
        ((-10, -8, 6), (10, -8, 6)),
        ((-10, 8, 6), (10, 8, 6)),
        ((-10, -8, -6), (-10, 8, -6)),
        ((10, -8, -6), (10, 8, -6)),
        ((-10, -8, 6), (-10, 8, 6)),
        ((10, -8, 6), (10, 8, 6)),
        ((-10, -8, -6), (-10, -8, 6)),
        ((10, -8, -6), (10, -8, 6)),
        ((-10, 8, -6), (-10, 8, 6)),
        ((10, 8, -6), (10, 8, 6)),
    ]
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertex)
    glEnd()

def main():
    pygame.init()
    pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
    gluPerspective(45, (WIDTH / HEIGHT), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -15)

    ball = Ball(INIT_POS, INIT_VELOCITY)

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                return

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        ball.move()
        ball.reflect_if_needed()

        draw_enclosure()
        draw_ball(ball)

        pygame.display.flip()
        pygame.time.wait(10)

if __name__ == "__main__":
    main()
