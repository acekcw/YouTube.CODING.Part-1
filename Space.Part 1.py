
from OpenGL.GL import *
from OpenGL.GLU import *
from enum import Enum

import glfw
import math
import random


DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600

gMouseXPos = 0


class Color(Enum):
    WHITE = (1.0, 1.0, 1.0)
    BLACK = (0.0, 0.0, 0.0)
    RED = (1.0, 0.0, 0.0)
    GREEN = (0.0, 1.0, 0.0)
    BLUE = (0.0, 0.0, 1.0)

class Star:
    def __init__(self):
        x = random.randrange(-500, 500)
        y = random.randrange(-500, 500)
        z = random.randrange(-500, 0)

        self.pos = [x, y, z]
        self.tailPos = [x, y, z]

    def GetPos(self):
        return self.pos

    def GetTailPos(self):
        return self.tailPos

    def Update(self, spacecraftSpeed):
        self.pos[2] += 1.0 * spacecraftSpeed
        #self.tailPos[2] += 1.0 * spacecraftSpeed * 0.95

        if self.pos[2] > 0.0:
            x = random.randrange(-500, 500)
            y = random.randrange(-500, 500)
            z = -500

            self.pos = [x, y, z]
            self.tailPos = [x, y, z]


def HandleCursorPosCallback(glfwWindow, xPos, yPos):
    global gMouseXPos

    gMouseXPos, yPos = glfw.get_cursor_pos(glfwWindow)

def GenerateEllipse(ellipse, xRadius, yRadius, numPoints):
    RADIAN = math.pi / 180.0

    deltaAngle = 360.0 / numPoints

    for i in range(numPoints):
        ellipse.append(math.cos(i * deltaAngle * RADIAN) * xRadius)
        ellipse.append(math.sin(i * deltaAngle * RADIAN) * yRadius)
        ellipse.append(0.0)

def DrawStars(stars, numStars, drawEllipse):
    ellipse = []
    lineEndPoint = []
    starPos = []
    starTailPos = []

    numEllipsePoints = 36

    GenerateEllipse(ellipse, 1.0, 1.0, numEllipsePoints)
    lineEndPoint = [0.0 for i in range(3)]

    spacecraftSpeed = (gMouseXPos / DISPLAY_WIDTH) * 50.0

    glColor(Color.WHITE.value)

    for i in range(numStars):
        stars[i].Update(spacecraftSpeed)

        starPos = stars[i].GetPos()
        starTailPos = stars[i].GetTailPos()

        lineEndPoint[0] = starTailPos[0] - starPos[0]
        lineEndPoint[1] = starTailPos[1] - starPos[1]
        lineEndPoint[2] = starTailPos[2] - starPos[2]

        glPushMatrix()

        glTranslatef(starPos[0], starPos[1], starPos[2])

        glBegin(GL_LINES)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(lineEndPoint[0], lineEndPoint[1], lineEndPoint[2])
        glEnd()

        glPopMatrix()

    if not drawEllipse:
        return

    glEnableClientState(GL_VERTEX_ARRAY)

    glVertexPointer(3, GL_FLOAT, 0, ellipse)    

    for i in range(numStars):
        #stars[i].Update(spacecraftSpeed)

        starPos = stars[i].GetPos()

        glPushMatrix()

        glTranslatef(starPos[0], starPos[1], starPos[2])

        glDrawArrays(GL_POLYGON, 0, numEllipsePoints)

        glPopMatrix()

    glDisableClientState(GL_VERTEX_ARRAY)

def Main():
    stars = []

    numStars = 400

    stars = [Star() for i in range(numStars)]

    fovy = 45.0
    aspect = DISPLAY_WIDTH / DISPLAY_HEIGHT
    near = 0.1
    far = 1000.0

    if not glfw.init():
        return

    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)

    glfwWindow = glfw.create_window(DISPLAY_WIDTH, DISPLAY_HEIGHT, 'Space.Part 1', None, None)

    if not glfwWindow:
        glfw.terminate()
        return

    videoMode = glfw.get_video_mode(glfw.get_primary_monitor())

    windowWidth = videoMode.size.width
    windowHeight = videoMode.size.height
    windowPosX = int(windowWidth / 2 - DISPLAY_WIDTH / 2) - 250
    windowPosY = int(windowHeight / 2 - DISPLAY_HEIGHT / 2) - 50

    glfw.set_window_pos(glfwWindow, windowPosX, windowPosY)

    glfw.show_window(glfwWindow)

    glfw.make_context_current(glfwWindow)

    glfw.set_cursor_pos_callback(glfwWindow, HandleCursorPosCallback)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(fovy, aspect, near, far)

    glClearColor(Color.BLACK.value[0], Color.BLACK.value[1], Color.BLACK.value[2], 1.0)

    while not glfw.window_should_close(glfwWindow):
        glfw.poll_events()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -10.0)

        DrawStars(stars, numStars, False)

        glfw.swap_buffers(glfwWindow)


glfw.terminate()


if __name__ == "__main__":
    Main()    