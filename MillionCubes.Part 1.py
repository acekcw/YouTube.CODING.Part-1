
from OpenGL.GL import *
from OpenGL.GL.shaders import *
from enum import Enum

import glfw
import glm
import math
import random

import numpy as np
import freetype as ft


gCubeVerticesData = [
    # Front
    -0.5, -0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0,
    0.5, -0.5, 0.5, 1.0, 0.0, 0.0, 1.0, 0.0,
    0.5, 0.5, 0.5, 1.0, 0.0, 0.0, 1.0, 1.0,
    -0.5, 0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0,

    # Back
    0.5, -0.5, -0.5, 0.0, 1.0, 0.0, 0.0, 0.0,
    -0.5, -0.5, -0.5, 0.0, 1.0, 0.0, 1.0, 0.0,
    -0.5, 0.5, -0.5, 0.0, 1.0, 0.0, 1.0, 1.0,
    0.5, 0.5, -0.5, 0.0, 1.0, 0.0, 0.0, 1.0,

    # Left
    -0.5, -0.5, -0.5, 0.0, 0.0, 1.0, 0.0, 0.0,
    -0.5, -0.5, 0.5, 0.0, 0.0, 1.0, 1.0, 0.0,
    -0.5, 0.5, 0.5, 0.0, 0.0, 1.0, 1.0, 1.0,
    -0.5, 0.5, -0.5, 0.0, 0.0, 1.0, 0.0, 1.0,

    # Right
    0.5, -0.5, 0.5, 1.0, 1.0, 0.0, 0.0, 0.0,
    0.5, -0.5, -0.5, 1.0, 1.0, 0.0, 1.0, 0.0,
    0.5, 0.5, -0.5, 1.0, 1.0, 0.0, 1.0, 1.0,
    0.5, 0.5, 0.5, 1.0, 1.0, 0.0, 0.0, 1.0,

    # Top
    -0.5, 0.5, 0.5, 0.0, 1.0, 1.0, 0.0, 0.0, 
    0.5, 0.5, 0.5, 0.0, 1.0, 1.0, 1.0, 0.0,
    0.5, 0.5, -0.5, 0.0, 1.0, 1.0, 1.0, 1.0,
    -0.5, 0.5, -0.5, 0.0, 1.0, 1.0, 0.0, 1.0,

    # Bottom
    -0.5, -0.5, -0.5, 1.0, 0.0, 1.0, 0.0, 0.0,
    0.5, -0.5, -0.5, 1.0, 0.0, 1.0, 1.0, 0.0,
    0.5, -0.5, 0.5, 1.0, 0.0, 1.0, 1.0, 1.0,
    -0.5, -0.5, 0.5, 1.0, 0.0, 1.0, 0.0, 1.0
    ]

gCubeIndicesData = [
    0, 1, 2, 2, 3, 0,
    4, 5, 6, 6, 7, 4,
    8, 9, 10, 10, 11, 8,
    12, 13, 14, 14, 15, 12,
    16, 17, 18, 18, 19, 16,
    20, 21, 22, 22, 23, 20
    ]


vertexShaderCode = """

# version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
layout(location = 3) in mat4 aInstanceMat;
layout(location = 7) in vec3 aInstanceColor;

out vec3 color;

uniform mat4 prjMat;
uniform mat4 viewMat;
uniform mat4 modelMat;

void main()
{
    color = aInstanceColor;

    gl_Position = prjMat * viewMat * aInstanceMat * modelMat * vec4(aPos, 1.0);
}

"""

fragmentShaderCode = """

# version 330 core

in vec3 color;

out vec4 fragColor;

void main()
{
    fragColor = vec4(color, 1.0);
}

"""


class Color(Enum):
    WHITE = (1.0, 1.0, 1.0)
    BLACK = (0.0, 0.0, 0.0)
    RED = (1.0, 0.0, 0.0)
    GREEN = (0.0, 1.0, 0.0)
    BLUE = (0.0, 0.0, 1.0)
    FPS_RED = (0.8, 0.3, 0.5)
    FPS_GREEN = (0.3, 0.8, 0.5)

class SceneManager:
    def __init__(self):
        self.displaySize = [800, 600]

        self.fovy = 45.0
        self.aspect = self.displaySize[0] / self.displaySize[1]
        self.near = 0.1
        self.far = 1000.0

        self.camera = None
        self.sailingCamera = False

        self.perspectivePrjMat = []
        self.orthoPrjMat = []
        self.viewMat = []

        self.meshes = []
        self.font = None

        self.dirty = True

    def GetDisplaySize(self):
        return self.displaySize

    def SetDisplaySize(self, width, height):
        self.displaySize[0] = width
        self.displaySize[1] = height
        self.aspect = self.displaySize[0] / self.displaySize[1]

        self.dirty = True

    def GetCamera(self):
        return self.camera

    def SetCamera(self, camera):
        self.camera = camera

    def GetPerspectivePrjMat(self):
        return self.perspectivePrjMat

    def GetOrthoPrjMat(self):
        return self.orthoPrjMat

    def GetViewMat(self):
        return self.viewMat

    def SetDirty(self, value):
        self.dirty = value

    def SetSailingCamera(self, value):
        self.sailingCamera = value

    def SetCameraPos(self):
        if gInputManager.GetKeyState('W') == True:
            self.camera.ProcessKeyboard('FORWARD', 0.05)
            self.dirty = True
        if gInputManager.GetKeyState('S') == True:
            self.camera.ProcessKeyboard('BACKWARD', 0.05)
            self.dirty = True
        if gInputManager.GetKeyState('A') == True:
            self.camera.ProcessKeyboard('LEFT', 0.05)
            self.dirty = True
        if gInputManager.GetKeyState('D') == True:
            self.camera.ProcessKeyboard('RIGHT', 0.05)
            self.dirty = True

    def SailCamera(self):
        self.camera.ProcessKeyboard('FORWARD', 1.0)
        self.dirty = True

    def InitializeOpenGL(self):
        glClearColor(Color.BLACK.value[0], Color.BLACK.value[1], Color.BLACK.value[2], 1.0)

        glEnable(GL_DEPTH_TEST)

        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)

    def MakeFont(self):
        self.font = Font('..\Fonts\comic.ttf', 24)
        self.font.MakeFontTextureWithGenList()

    def AddMesh(self, mesh):
        self.meshes.append(mesh)

    def UpdateAboutInput(self):
        if gInputManager.GetKeyState('1') == True:
            self.sailingCamera = not self.sailingCamera
            gInputManager.SetKeyState('1', False)

    def Update(self, deltaTime):
        numMeshes = len(self.meshes)

        for i in range(numMeshes):
            self.meshes[i].Update(deltaTime)

        self.UpdateAboutInput()

        self.SetCameraPos()

        if self.sailingCamera == True:
            self.SailCamera()

        if self.dirty == False:
            return

        self.perspectivePrjMat = glm.perspective(self.fovy, self.aspect, self.near, self.far)

        self.orthoPrjMat = glm.ortho(0, self.displaySize[0], 0, self.displaySize[1], -1.0, 1.0)

        self.viewMat = self.camera.GetViewMat()

        self.dirty = False

    def DrawMeshes(self):
        numMeshes = len(self.meshes)

        for i in range(numMeshes):
            self.meshes[i].Draw()

    def DrawText(self, x, y, text, color):
        glPushAttrib(GL_COLOR_BUFFER_BIT)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()

        glLoadIdentity()

        glOrtho(0, self.displaySize[0], 0, self.displaySize[1], -1.0, 1.0)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()

        glLoadIdentity()

        glTranslatef(x, y, 0.0)

        texId = self.font.GetTexId()

        glBindTexture(GL_TEXTURE_2D, texId)

        glColor(color)

        glListBase(self.font.GetListOffset())
        glCallLists([ord(c) for c in text])

        glPopMatrix()

        glMatrixMode(GL_PROJECTION)

        glPopMatrix()

        glPopAttrib()

class InputManager:
    def __init__(self):
        self.mouseEntered = False

        self.lastMousePos = [-1, 1]

        self.keys = {}

    def GetMouseEntered(self):
        return self.mouseEntered

    def SetMouseEntered(self, value):
        self.mouseEntered = value

    def GetLastMousePos(self):
        return self.lastMousePos

    def SetLastMousePos(self, value):
        self.lastMousePos = value

    def GetKeyState(self, key):
        if key in self.keys.keys():
            return self.keys[key]

    def SetKeyState(self, key, value):
        self.keys[key] = value

class Camera:
    def __init__(self):
        self.cameraPos = glm.vec3(0.0, 0.0, 300.0)
        self.cameraFront = glm.vec3(0.0, 0.0, -1.0)
        self.cameraUp = glm.vec3(0.0, 1.0, 0.0)
        self.cameraRight = glm.vec3(1.0, 0.0, 0.0)
        self.cameraWorldUp = glm.vec3(0.0, 1.0, 0.0)

        self.pitch = 0.0
        self.yaw = 180.0

        self.mouseSensitivity = 0.1

        self.UpdateCameraVectors()

    def GetViewMat(self):
        return glm.lookAt(self.cameraPos, self.cameraPos + self.cameraFront, self.cameraUp)

    def ProcessMouseMovement(self, xOffset, yOffset, constrainPitch = True):
        xOffset *= self.mouseSensitivity
        yOffset *= self.mouseSensitivity

        self.yaw += xOffset
        self.pitch += yOffset

        if constrainPitch == True:
            if self.pitch > 89.0:
                self.pitch = 89.0
            elif self.pitch < -89.0:
                self.pitch = -89.0

        self.UpdateCameraVectors()

    def ProcessKeyboard(self, direction, velocity):
        if direction == "FORWARD":
            self.cameraPos += self.cameraFront * velocity
        elif direction == "BACKWARD":
            self.cameraPos -= self.cameraFront * velocity
        elif direction == "LEFT":
            self.cameraPos += self.cameraRight * velocity
        elif direction == "RIGHT":
            self.cameraPos -= self.cameraRight * velocity

    def UpdateCameraVectors(self):
        self.cameraFront.x = math.sin(glm.radians(self.yaw)) * math.cos(glm.radians(self.pitch))
        self.cameraFront.y = math.sin(glm.radians(self.pitch))
        self.cameraFront.z = math.cos(glm.radians(self.yaw)) * math.cos(glm.radians(self.pitch))

        self.cameraFront = glm.normalize(self.cameraFront)

        self.cameraRight = glm.normalize(glm.cross(self.cameraWorldUp, self.cameraFront))
        self.cameraUp = glm.normalize(glm.cross(self.cameraFront, self.cameraRight))

class Mesh:
    def __init__(self, shader, *datas):
        self.vertices = np.array(datas[0], dtype = np.float32)
        self.indices = np.array(datas[1], dtype = np.uint32)

        self.instanceTotalModelMat = []
        self.modelMat = []

        self.VAO = glGenVertexArrays(1)

        self.shader = shader

        VBO = glGenBuffers(1)
        EBO = glGenBuffers(1)

        glBindVertexArray(self.VAO)

        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.vertices.itemsize * 8, ctypes.c_void_p(0))

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, self.vertices.itemsize * 8, ctypes.c_void_p(self.vertices.itemsize * 3))

        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, self.vertices.itemsize * 8, ctypes.c_void_p(self.vertices.itemsize * 6))

        self._SetInstance()

    def Update(self, deltaTime):
        rotXMat = glm.rotate(deltaTime, glm.vec3(1.0, 0.0, 0.0))
        rotYMat = glm.rotate(deltaTime, glm.vec3(0.0, 1.0, 0.0))
        rotZMat = glm.rotate(deltaTime, glm.vec3(0.0, 0.0, 1.0))

        self.modelMat = rotZMat * rotYMat * rotXMat        

    def Draw(self):
        self.shader.SetMat4('modelMat', self.modelMat)

        glBindVertexArray(self.VAO)        
        glDrawElementsInstanced(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None, len(self.instanceTotalModelMat))
        glBindVertexArray(0)

    def _SetInstance(self):
        instanceModelMat = []
        instanceTotalColor = []

        xMax, yMax, zMax = 100, 100, 100
        x, y, z = 0, 0, 0

        xInterval, yInterval, zInterval = 2, 2, 2
        xOffset, yOffset, zOffset = (xMax - 1) * xInterval / 2, (yMax - 1) * yInterval / 2, (zMax - 1) * zInterval / 2

        for d in range(0, zMax):
            for r in range(0, yMax):
                for c in range(0, xMax):
                    x = c * xInterval - xOffset
                    y = r * yInterval - yOffset
                    z = d * zInterval - zOffset

                    instanceModelMat = glm.translate(glm.vec3(x, y, z))
                    self.instanceTotalModelMat.append(np.transpose(np.array(instanceModelMat)))

                    instanceTotalColor.append([random.random(), random.random(), random.random()])

        instanceTotalModelMat1D = np.array(self.instanceTotalModelMat, np.float32).flatten()
        instanceTotalColor1D = np.array(instanceTotalColor, np.float32).flatten()

        instanceTotalModelMatSize = len(self.instanceTotalModelMat)
        instanceModelMatBytes = int(instanceTotalModelMat1D.nbytes / instanceTotalModelMatSize)
        instanceModelMatDivideBytes = int(instanceModelMatBytes / 4)

        instanceVBO = glGenBuffers(2)
        glBindBuffer(GL_ARRAY_BUFFER, instanceVBO[0])
        glBufferData(GL_ARRAY_BUFFER, instanceTotalModelMat1D.nbytes, instanceTotalModelMat1D, GL_STATIC_DRAW)

        glEnableVertexAttribArray(3)
        glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, instanceModelMatBytes, ctypes.c_void_p(0))
        glEnableVertexAttribArray(4)
        glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, instanceModelMatBytes, ctypes.c_void_p(instanceModelMatDivideBytes))
        glEnableVertexAttribArray(5)
        glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, instanceModelMatBytes, ctypes.c_void_p(instanceModelMatDivideBytes * 2))
        glEnableVertexAttribArray(6)
        glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, instanceModelMatBytes, ctypes.c_void_p(instanceModelMatDivideBytes * 3))
                
        glVertexAttribDivisor(3, 1)
        glVertexAttribDivisor(4, 1)
        glVertexAttribDivisor(5, 1)
        glVertexAttribDivisor(6, 1)

        glBindBuffer(GL_ARRAY_BUFFER, instanceVBO[1])
        glBufferData(GL_ARRAY_BUFFER, instanceTotalColor1D.nbytes, instanceTotalColor1D, GL_STATIC_DRAW)

        glEnableVertexAttribArray(7)
        glVertexAttribPointer(7, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

        glVertexAttribDivisor(7, 1)

class Shader:
    def __init__(self, vsCode, fsCode):
        self.program = None

        self.program = compileProgram(compileShader(vsCode, GL_VERTEX_SHADER), compileShader(fsCode, GL_FRAGMENT_SHADER))

    def Use(self):
        glUseProgram(self.program)

    def SetMat4(self, name, value):
        loc = glGetUniformLocation(self.program, name)

        value = np.array(value, dtype = np.float32)
        glUniformMatrix4fv(loc, 1, GL_TRUE, value)

class Font:
    def __init__(self, fontName, size):
        self.face = ft.Face(fontName)
        self.face.set_char_size(size << 6)

        self.charsSize = (6, 16)
        self.charsAdvanceX = []

        self.maxCharHeight = 0
        self.charStartOffset = 32
        self.listOffset = -1
        self.texId = -1

        numChars = self.charsSize[0] * self.charsSize[1]

        self.charsAdvanceX =  [0 for i in range(numChars)]

        advanceX, ascender, descender = 0, 0, 0
        charEndIndex = self.charStartOffset + numChars

        for c in range(self.charStartOffset, charEndIndex):
            self.face.load_char(chr(c), ft.FT_LOAD_RENDER | ft.FT_LOAD_FORCE_AUTOHINT)

            self.charsAdvanceX[c - self.charStartOffset] = self.face.glyph.advance.x >> 6

            advanceX = max(advanceX, self.face.glyph.advance.x >> 6)
            ascender = max(ascender, self.face.glyph.metrics.horiBearingY >> 6)
            descender = max(descender, (self.face.glyph.metrics.height >> 6) - (self.face.glyph.metrics.horiBearingY >> 6))

        self.maxCharHeight = ascender + descender
        maxTotalAdvanceX = advanceX * self.charsSize[1]
        maxTotalHeight = self.maxCharHeight * self.charsSize[0]

        exponent = 0
        bitmapDataSize = [0, 0]

        while maxTotalAdvanceX > math.pow(2, exponent):
            exponent += 1
        bitmapDataSize[1] = int(math.pow(2, exponent))

        exponent = 0

        while maxTotalHeight > math.pow(2, exponent):
            exponent += 1 
        bitmapDataSize[0] = int(math.pow(2, exponent))

        self.bitmapData = np.zeros((bitmapDataSize[0], bitmapDataSize[1]), dtype = np.ubyte)

        x, y, charIndex = 0, 0, 0

        for r in range(self.charsSize[0]):
            for c in range(self.charsSize[1]):
                self.face.load_char(chr(self.charStartOffset + r * self.charsSize[1] + c), ft.FT_LOAD_RENDER | ft.FT_LOAD_FORCE_AUTOHINT)

                charIndex = r * self.charsSize[1] + c

                bitmap = self.face.glyph.bitmap
                x += self.face.glyph.bitmap_left
                y = r * self.maxCharHeight + ascender - self.face.glyph.bitmap_top

                self.bitmapData[y : y + bitmap.rows, x : x + bitmap.width].flat = bitmap.buffer

                x += self.charsAdvanceX[charIndex] - self.face.glyph.bitmap_left

            x = 0

    def GetTexId(self):
        return self.texId

    def GetListOffset(self):
        return self.listOffset

    def MakeFontTextureWithGenList(self):
        self.texId = glGenTextures(1)

        glBindTexture(GL_TEXTURE_2D, self.texId)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)

        self.bitmapData = np.flipud(self.bitmapData)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_ALPHA, self.bitmapData.shape[1], self.bitmapData.shape[0], 0, 
                     GL_ALPHA, GL_UNSIGNED_BYTE, self.bitmapData)

        dx = 0.0
        dy = self.maxCharHeight / float(self.bitmapData.shape[0])

        listStartIndex = glGenLists(self.charsSize[0] * self.charsSize[1])
        self.listOffset = listStartIndex - self.charStartOffset

        for r in range(self.charsSize[0]):
            for c in range(self.charsSize[1]):
                glNewList(listStartIndex + r * self.charsSize[1] + c, GL_COMPILE)

                charIndex = r * self.charsSize[1] + c

                advanceX = self.charsAdvanceX[charIndex]
                dAdvanceX = advanceX / float(self.bitmapData.shape[1])

                glBegin(GL_QUADS)
                glTexCoord2f(dx, 1.0 - r * dy), glVertex3f(0.0, 0.0, 0.0)
                glTexCoord2f(dx + dAdvanceX, 1.0 - r * dy), glVertex3f(advanceX, 0.0, 0.0)
                glTexCoord2f(dx + dAdvanceX, 1.0 - (r + 1) * dy), glVertex3f(advanceX, -self.maxCharHeight, 0.0)
                glTexCoord2f(dx, 1.0 - (r + 1) * dy), glVertex3f(0.0, -self.maxCharHeight, 0.0)
                glEnd()

                glTranslatef(advanceX, 0.0, 0.0)

                glEndList()

                dx += dAdvanceX

            glTranslatef(0.0, -self.maxCharHeight, 0.0)
            dx = 0.0

gSceneManager = SceneManager()
gInputManager = InputManager()

def HandleWindowSizeCallback(glfwWindow, width, height):
    glViewport(0, 0, width, height)

    gSceneManager.SetDisplaySize(width, height)

def HandleKeyCallback(glfwWindow, key, scanCode, action, modes):
    if action == glfw.PRESS:
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(glfwWindow, glfw.TRUE)

        if key == glfw.KEY_1:
            gInputManager.SetKeyState('1', True)

        if key == glfw.KEY_W:
            gInputManager.SetKeyState('W', True)
        elif key == glfw.KEY_S:
            gInputManager.SetKeyState('S', True)
        elif key == glfw.KEY_A:
            gInputManager.SetKeyState('A', True)
        elif key == glfw.KEY_D:
            gInputManager.SetKeyState('D', True)

    if action == glfw.RELEASE:
        if key == glfw.KEY_W:
            gInputManager.SetKeyState('W', False)
        elif key == glfw.KEY_S:
            gInputManager.SetKeyState('S', False)
        elif key == glfw.KEY_A:
            gInputManager.SetKeyState('A', False)
        elif key == glfw.KEY_D:
            gInputManager.SetKeyState('D', False)

def HandleCursorPosCallback(glfwWindow, xPos, yPos):
    if gInputManager.GetMouseEntered() == False:
        gInputManager.SetLastMousePos([xPos, yPos])
        gInputManager.SetMouseEntered(True)

    lastPos = gInputManager.GetLastMousePos()
    xOffset = lastPos[0] - xPos
    yOffset = lastPos[1] - yPos

    gInputManager.SetLastMousePos([xPos, yPos])

    camera = gSceneManager.GetCamera()
    camera.ProcessMouseMovement(xOffset, yOffset)

    displaySize = gSceneManager.GetDisplaySize()

    mouseCheckInterval = 10

    if xPos < mouseCheckInterval:
        glfw.set_cursor_pos(glfwWindow, displaySize[0] - mouseCheckInterval, yPos)
        gInputManager.SetMouseEntered(False)
    elif xPos > displaySize[0] - mouseCheckInterval:
        glfw.set_cursor_pos(glfwWindow, mouseCheckInterval, yPos)
        gInputManager.SetMouseEntered(False)

    if yPos < mouseCheckInterval:
        glfw.set_cursor_pos(glfwWindow, xPos, displaySize[1] - mouseCheckInterval)
        gInputManager.SetMouseEntered(False)
    elif yPos > displaySize[1] - mouseCheckInterval:
        glfw.set_cursor_pos(glfwWindow, xPos, mouseCheckInterval)
        gInputManager.SetMouseEntered(False)

    gSceneManager.SetDirty(True)

def Main():
    displaySize = gSceneManager.GetDisplaySize()

    if not glfw.init():
        return

    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)

    glfwWindow = glfw.create_window(displaySize[0], displaySize[1], 'MillionCubes.Part 1', None, None)

    if not glfwWindow:
        glfw.terminate()
        return

    videoMode = glfw.get_video_mode(glfw.get_primary_monitor())

    windowWidth = videoMode.size.width
    windowHeight = videoMode.size.height
    windowPosX = int(windowWidth / 2 - displaySize[0] / 2) - 250
    windowPosY = int(windowHeight / 2 - displaySize[1] / 2) - 50

    glfw.set_window_pos(glfwWindow, windowPosX, windowPosY)

    glfw.show_window(glfwWindow)

    glfw.set_input_mode(glfwWindow, glfw.CURSOR, glfw.CURSOR_DISABLED)

    glfw.make_context_current(glfwWindow)

    glfw.set_window_size_callback(glfwWindow, HandleWindowSizeCallback)

    glfw.set_key_callback(glfwWindow, HandleKeyCallback)

    glfw.set_cursor_pos_callback(glfwWindow, HandleCursorPosCallback)

    gSceneManager.InitializeOpenGL()
    gSceneManager.SetCamera(Camera())
    gSceneManager.MakeFont()    

    shader = Shader(vertexShaderCode, fragmentShaderCode)

    cubeMesh = Mesh(shader, gCubeVerticesData, gCubeIndicesData)

    gSceneManager.AddMesh(cubeMesh)    

    prjMat = []
    viewMat = []

    gSceneManager.MakeFont()

    lastElapsedTime = glfw.get_time()

    while glfw.window_should_close(glfwWindow) == False:
        glfw.poll_events()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        gSceneManager.Update(lastElapsedTime)

        prjMat = gSceneManager.GetPerspectivePrjMat()
        viewMat = gSceneManager.GetViewMat()

        shader.Use()

        shader.SetMat4('prjMat', prjMat)
        shader.SetMat4('viewMat', viewMat)

        gSceneManager.DrawMeshes()

        glUseProgram(0)

        gSceneManager.DrawText(660, 590, 'FPS: {0: 0.2f}'.format(1.0 / (glfw.get_time() - lastElapsedTime)), Color.FPS_RED.value)        

        glfw.swap_buffers(glfwWindow)

        lastElapsedTime = glfw.get_time()


glfw.terminate()


if __name__ == "__main__":
    Main()