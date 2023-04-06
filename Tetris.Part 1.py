
from OpenGL.GL import *
from OpenGL.GL.shaders import *
from enum import Enum

import glfw
import glm
import math
import random

import numpy as np
import freetype as ft


vertexShaderCode = """

# version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec4 aColor;

out vec4 color;

uniform mat4 prjMat;
uniform mat4 viewMat;
uniform mat4 modelMat;

void main()
{
    color = aColor;

    gl_Position = prjMat * viewMat * vec4(aPos, 1.0);
}

"""

fragmentShaderCode = """

# version 330 core

in vec4 color;

out vec4 fragColor;

void main()
{
    fragColor = color;
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
    FPS_BLUE = (0.2, 0.3, 0.98)

class SceneManager:
    def __init__(self, view3D):
        self.displaySize = [800, 600]

        self.programInfoAreaVertices = []
        self.programInfoAreaIndices = []

        self.fovy = 45.0
        self.aspect = self.displaySize[0] / self.displaySize[1]
        self.near = 0.1
        self.far = 1000.0

        self.camera = None
        self.sailingCamera = [False, False]

        self.perspectivePrjMat = glm.mat4()
        self.orthoPrjMat = glm.mat4()
        self.viewMat = glm.mat4()
        self.view3D = view3D

        self.objects = []
        self.font = None

        self.deltaTime = 0.0
        self.dirty = True

        self.programInfo = True
        self.numProgramInfoElement = 5

        self.pause = False
        self.debug = False
        self.debugMat = glm.mat4()

        self._InitializeProgramInfoArea()

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

    def GetPause(self):
        return self.pause

    def GetView3D(self):
        return self.view3D

    def SetDirty(self, value):
        self.dirty = value

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
        if self.sailingCamera[0] == True:
            self.camera.ProcessKeyboard('FORWARD', 1.0)
            self.dirty = True
        if self.sailingCamera[1] == True:
            self.camera.ProcessKeyboard('BACKWARD', 1.0)
            self.dirty = True

    def InitializeOpenGL(self):
        glClearColor(Color.BLACK.value[0], Color.BLACK.value[1], Color.BLACK.value[2], 1.0)

        glEnable(GL_DEPTH_TEST)

    def MakeFont(self):
        self.font = Font('..\Fonts\comic.ttf', 14)
        self.font.MakeFontTextureWithGenList()

    def AddObject(self, object):
        self.objects.append(object)

    def UpdateAboutInput(self):
        numObjects = len(self.objects)

        if gInputManager.GetKeyState(glfw.KEY_SPACE) == True:
            for i in range(numObjects):
                self.objects[i].UpdateAboutInput(glfw.KEY_SPACE)
            gInputManager.SetKeyState(glfw.KEY_SPACE, False)

        if gInputManager.GetKeyState('1') == True:
            self.sailingCamera[0] = not self.sailingCamera[0]
            gInputManager.SetKeyState('1', False)
        if gInputManager.GetKeyState('2') == True:
            self.sailingCamera[1] = not self.sailingCamera[1]
            gInputManager.SetKeyState('2', False)

        if gInputManager.GetKeyState('B') == True:            
            self.debug = not self.debug
            gInputManager.SetKeyState('B', False)
        if gInputManager.GetKeyState('I') == True:
            self.programInfo = not self.programInfo                
            gInputManager.SetKeyState('I', False)
        if gInputManager.GetKeyState('P') == True:
            self.pause = not self.pause            
            gInputManager.SetKeyState('P', False)
        if gInputManager.GetKeyState('R') == True:
            for i in range(numObjects):
                self.objects[i].Restart()
            gInputManager.SetKeyState('R', False)

        if gInputManager.GetKeyState(glfw.KEY_UP) == True:
            for i in range(numObjects):
                self.objects[i].UpdateAboutInput(glfw.KEY_UP)
            gInputManager.SetKeyState(glfw.KEY_UP, False)
        elif gInputManager.GetKeyState(glfw.KEY_DOWN) == True:
            for i in range(numObjects):
                self.objects[i].UpdateAboutInput(glfw.KEY_DOWN)
            gInputManager.SetKeyState(glfw.KEY_DOWN, False)
        elif gInputManager.GetKeyState(glfw.KEY_LEFT) == True:
            for i in range(numObjects):
                self.objects[i].UpdateAboutInput(glfw.KEY_LEFT)
            gInputManager.SetKeyState(glfw.KEY_LEFT, False)
        elif gInputManager.GetKeyState(glfw.KEY_RIGHT) == True:
            for i in range(numObjects):
                self.objects[i].UpdateAboutInput(glfw.KEY_RIGHT)
            gInputManager.SetKeyState(glfw.KEY_RIGHT, False)

    def Update(self, deltaTime):
        self.UpdateAboutInput()

        if self.pause == True:
            return

        numObjects = len(self.objects)

        for i in range(numObjects):
            self.objects[i].Update(deltaTime)        

        if self.view3D == True:
            self.SetCameraPos()
            self.SailCamera()

        if self.dirty == False:
            return

        self.perspectivePrjMat = glm.perspective(self.fovy, self.aspect, self.near, self.far)

        self.orthoPrjMat = glm.ortho(0, self.displaySize[0], 0, self.displaySize[1], 1.0, 100.0)

        self.viewMat = self.camera.GetViewMat()

        self.deltaTime += deltaTime
        self.dirty = False

    def DrawObjects(self):
        numObjects = len(self.objects)        

        for i in range(numObjects):
            self.objects[i].Draw()

    def DrawProgramInfo(self, deltaTime):
        if self.programInfo == False:
            return

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()

        glOrtho(0, self.displaySize[0], 0, self.displaySize[1], -10.0, 10.0)        

        glMatrixMode(GL_MODELVIEW)        

        #self.debugMat = glGetFloatv(GL_MODELVIEW_MATRIX)

        self._DrawProgramInfoArea()

        self._DrawProgramInfo(deltaTime)

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()

    def _InitializeProgramInfoArea(self):
        programInfoAreaVerticesData = [
            585.0, 590.0, -4.5, 1.0, 1.0, 1.0, 1.0,
            785.0, 590.0, -4.5, 1.0, 1.0, 1.0, 1.0,
            585.0, 587.0, -4.5, 1.0, 1.0, 1.0, 1.0,
            785.0, 587.0, -4.5, 1.0, 1.0, 1.0, 1.0,

            580.0, 570.0, -4.5, 0.0, 0.0, 1.0, 0.8,
            580.0, 370.0, -4.5, 0.0, 0.0, 1.0, 0.8,
            577.0, 570.0, -4.5, 0.0, 0.0, 1.0, 0.8,
            577.0, 370.0, -4.5, 0.0, 0.0, 1.0, 0.8,

            585.0, 353.0, -4.5, 0.0, 0.0, 1.0, 0.8,
            785.0, 353.0, -4.5, 0.0, 0.0, 1.0, 0.8,
            585.0, 350.0, -4.5, 0.0, 0.0, 1.0, 0.8,
            785.0, 350.0, -4.5, 0.0, 0.0, 1.0, 0.8,

            790.0, 570.0, -4.5, 0.0, 0.0, 1.0, 0.8,
            790.0, 370.0, -4.5, 0.0, 0.0, 1.0, 0.8,
            793.0, 570.0, -4.5, 0.0, 0.0, 1.0, 0.8,
            793.0, 370.0, -4.5, 0.0, 0.0, 1.0, 0.8
            ]
        
        programInfoAreaIndicesData = [
            0, 1,
            2, 3,

            4, 5,
            6, 7,

            8, 9,
            10, 11,

            12, 13,
            14, 15,
            ]

        self.programInfoAreaVertices = np.array(programInfoAreaVerticesData, dtype = np.float32)
        self.programInfoAreaIndices = np.array(programInfoAreaIndicesData, dtype = np.uint32)
       
    def _DrawProgramInfoArea(self):
        glPushMatrix()
        glLoadIdentity()

        glPushAttrib(GL_COLOR_BUFFER_BIT | GL_ENABLE_BIT | GL_LINE_BIT)

        glDisable(GL_DEPTH_TEST)

        glEnable(GL_BLEND)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)        
        
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        
        glVertexPointer(3, GL_FLOAT, self.programInfoAreaVertices.itemsize * 7,  ctypes.c_void_p(self.programInfoAreaVertices.ctypes.data))
        glColorPointer(4, GL_FLOAT, self.programInfoAreaVertices.itemsize * 7,  ctypes.c_void_p(self.programInfoAreaVertices.ctypes.data + self.programInfoAreaVertices.itemsize * 3))

        glDrawElements(GL_LINES, len(self.programInfoAreaIndices), GL_UNSIGNED_INT, self.programInfoAreaIndices)

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

        glPopAttrib()

        glPopMatrix()
       
    def _DrawProgramInfo(self, deltaTime):
        glPushMatrix()
        glLoadIdentity()

        glPushAttrib(GL_COLOR_BUFFER_BIT | GL_ENABLE_BIT)

        glDisable(GL_DEPTH_TEST)

        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        texId = self.font.GetTexId()

        glBindTexture(GL_TEXTURE_2D, texId)

        glColor(Color.FPS_GREEN.value)        

        infoText = []
        infoTextIndex = 0

        infoText.append('.FPS: {0: 0.2f}'.format(0.0))

        if deltaTime != 0.0:
            infoText[infoTextIndex] = ".FPS: {0: 0.2f}".format(1.0 / deltaTime)

        infoText.append('.ViewMode: ')
        infoTextIndex += 1

        if self.view3D == True:
            infoText[infoTextIndex] += "3D"
        else:
            infoText[infoTextIndex] += "2D"

        infoText.append('.SailingDir(1, 2): ')
        infoTextIndex += 1

        if self.sailingCamera[0] == True:
            infoText[infoTextIndex] += "F"
        if self.sailingCamera[1] == True:
            infoText[infoTextIndex] += "B"

        infoText.append('.Pause(P): ')
        infoTextIndex += 1
        
        if self.pause == True:
            infoText[infoTextIndex] += "On"
        else:
            infoText[infoTextIndex] += "Off"

        infoText.append('.Debug(B): ')
        infoTextIndex += 1
        
        if self.debug == True:
            infoText[infoTextIndex] += "On"
        else:
            infoText[infoTextIndex] += "Off"

        textPosX = 590.0
        textPosY = 570.0

        for i in range(self.numProgramInfoElement):
            glTranslate(textPosX, textPosY, 0.0)

            glListBase(self.font.GetListOffset())
            glCallLists([ord(c) for c in infoText[i]])        

            glPopMatrix()
            glPushMatrix()
            glLoadIdentity()

            textPosY -= 20.0

        glPopAttrib()

        glPopMatrix()

class InputManager:
    def __init__(self):
        self.mouseEntered = False

        self.lastMousePos = [-1, -1]

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
        self.cameraPos = glm.vec3(0.0, 0.0, 10.0)
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
    def __init__(self, shader, dataType = -1):
        self.vertices = []
        self.indices = []

        self.modelMat = glm.mat4()

        self.rotDegree = 0.0

        self.shader = shader
        self.dataType = dataType

        if self.dataType == -1:
            self._GenerateCube()

        self.VAO = glGenVertexArrays(1)

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

        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glBindVertexArray(0)

    def Restart(self):
        return

    def Update(self, deltaTime):
        rotRadian = glm.radians(self.rotDegree)

        rotXMat = glm.rotate(rotRadian, glm.vec3(1.0, 0.0, 0.0))
        rotYMat = glm.rotate(rotRadian, glm.vec3(0.0, 1.0, 0.0))
        rotZMat = glm.rotate(rotRadian, glm.vec3(0.0, 0.0, 1.0))

        self.modelMat = rotZMat * rotYMat * rotXMat
        #self.modelMat = rotYMat
        #self.modelMat = glm.mat4()

        self.rotDegree += deltaTime * 50

        if self.rotDegree > 360.0:
            self.rotDegree = 0.0

    def Draw(self):
        self.shader.SetMat4('modelMat', self.modelMat)

        glBindVertexArray(self.VAO)
        glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

    def _GenerateCube(self):
        cubeVerticesData = [
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

        cubeIndicesData = [
            0, 1, 2, 2, 3, 0,
            4, 5, 6, 6, 7, 4,
            8, 9, 10, 10, 11, 8,
            12, 13, 14, 14, 15, 12,
            16, 17, 18, 18, 19, 16,
            20, 21, 22, 22, 23, 20
            ]

        self.vertices = np.array(cubeVerticesData, dtype = np.float32)
        self.indices = np.array(cubeIndicesData, dtype = np.uint32)

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

        self.charsAdvanceX = [0 for i in range(numChars)]

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

                glTranslate(advanceX, 0.0, 0.0)

                glEndList()

                dx += dAdvanceX

            glTranslatef(0.0, -self.maxCharHeight, 0.0)
            dx = 0.0

class Tetris:
    def __init__(self, displaySize):
        self.backgroundStuffVerticesList = []
        self.backgroundStuffIndicesList = []

        self.backgroundVertices = []
        self.backgroundIndices = []

        self.backgroundSubAreaVertices = []
        self.backgroundSubAreaIndices = []

        self.backgroundLineVertices = []
        self.backgroundLineIndices = []

        self.backgroundSubAreaLineVertices = []
        self.backgroundSubAreaLineIndices = []

        self.backgroundBoardLineVertices = []
        self.backgroundBoardLineIndices = []

        self.gameStuffVerticesList = []
        self.gameStuffIndicesList = []

        self.allBlocksVertices = []
        self.allBlocksIndices = []
        
        self.preShapeVertices = []
        self.preShapeIndices = []

        self.curShapeVertices = []
        self.curShapeIndices = []

        self.shapeDistributionVertices = []
        self.shapeDistributionIndices = []

        self.boardSize = (10, 22)
        self.boardPos = []

        self.allBlocks = []

        self.allShapes = []
        self.allShapeColors = []
        self.allShapeDistribution = []

        self.preShape = []
        self.curShape = []

        self.preShapeIdx = -1
        self.curShapeIdx = -1

        self.startReferenceBoardPos = (5, 22)       

        self.displaySize = (displaySize[0], displaySize[1])

        self.numShapes = 7
        self.numBlocksOneShape = 4
        self.numVerticesOneBlock = 4
        self.numBackgroundStuff = 5
        self.numGameStuff = 4

        self.numVertexComponents = 7

        self.blockSize = 25       

        self.backgroundVAO = glGenVertexArrays(self.numBackgroundStuff)
        self.backgroundVBO = glGenBuffers(self.numBackgroundStuff)
        self.backgroundEBO = glGenBuffers(self.numBackgroundStuff)

        self.gameVAO = glGenVertexArrays(self.numGameStuff)
        self.gameVBO = glGenBuffers(self.numGameStuff)
        self.gameEBO = glGenBuffers(self.numGameStuff)

        self.shapeDownMaxTime = 0.5
        self.shapeDownAccumulatedTime = 0.0

        self.move = [False, False, False]
        self.rotate = False
        self.stackShapeImmediately = False

        self._InitializeBackground()

        self._InitializeShape()

    def Restart(self):
        self.allBlocks = np.full((self.boardSize[1], self.boardSize[0]), fill_value = 255)

        self._InitializeShape()

    def UpdateAboutInput(self, key):
        if key == glfw.KEY_SPACE:
            self.stackShapeImmediately = True

        if key == glfw.KEY_UP:
            self.rotate = True
        elif key == glfw.KEY_DOWN:
            self.move[2] = True
        elif key == glfw.KEY_LEFT:
            self.move[0] = True
        elif key == glfw.KEY_RIGHT:
            self.move[1] = True

    def Update(self, deltaTime):
        self._UpdateShape()

        self.shapeDownAccumulatedTime += deltaTime

        if self.shapeDownMaxTime < self.shapeDownAccumulatedTime:
            self.move[2] = True
            self.shapeDownAccumulatedTime = 0.0

    def Draw(self):
        self._DrawBackground()

        self._DrawStackBlocks()

        self._DrawShape()

        self._DrawPreShape()

        self._DrawShapeDistribution()

    def _CreateNewShape(self):
        self.preShape.clear()
        self.curShape.clear()

        self.curShapeIdx = self.preShapeIdx
        self.preShapeIdx = random.randrange(0, self.numShapes)

        for i in range(self.numBlocksOneShape):
            prePosX = int(self.allShapes[self.preShapeIdx][i] % 4)
            prePosY = int(self.allShapes[self.preShapeIdx][i] / 4)

            self.preShape.append([prePosX, prePosY])

            boardPosX = self.startReferenceBoardPos[0] + int(self.allShapes[self.curShapeIdx][i] % 4)
            boardPosY = self.startReferenceBoardPos[1] - int(self.allShapes[self.curShapeIdx][i] / 4)
            
            self.curShape.append([boardPosX, boardPosY])

        preShapeColor = self.allShapeColors[self.preShapeIdx]
        curShapeColor = self.allShapeColors[self.curShapeIdx]

        for i in range(self.numBlocksOneShape):
            iOffset = i * self.numVerticesOneBlock * self.numVertexComponents

            for j in range(self.numVerticesOneBlock):
                jOffset = j * self.numVertexComponents

                self.preShapeVertices[iOffset + jOffset + 3] = preShapeColor[0]
                self.preShapeVertices[iOffset + jOffset + 4] = preShapeColor[1]
                self.preShapeVertices[iOffset + jOffset + 5] = preShapeColor[2]
                self.preShapeVertices[iOffset + jOffset + 6] = 0.8

                self.curShapeVertices[iOffset + jOffset + 3] = curShapeColor[0]
                self.curShapeVertices[iOffset + jOffset + 4] = curShapeColor[1]
                self.curShapeVertices[iOffset + jOffset + 5] = curShapeColor[2]
                self.curShapeVertices[iOffset + jOffset + 6] = 0.8

        glBindBuffer(GL_ARRAY_BUFFER, self.gameVBO[1])
        glBufferSubData(GL_ARRAY_BUFFER, 0, self.gameStuffVerticesList[1].nbytes, self.gameStuffVerticesList[1])

        glBindBuffer(GL_ARRAY_BUFFER, self.gameVBO[2])
        glBufferSubData(GL_ARRAY_BUFFER, 0, self.gameStuffVerticesList[2].nbytes, self.gameStuffVerticesList[2])

        glBindBuffer(GL_ARRAY_BUFFER, 0)    

    def _InitializeBackground(self):
        self.allBlocks = np.full((self.boardSize[1], self.boardSize[0]), fill_value = 255)

        boardLbPos = [200.0, 50.0]
        boardRtPos = []
        boardRtPos.append(boardLbPos[0] + self.boardSize[0] * self.blockSize)
        boardRtPos.append(boardLbPos[1] + (self.boardSize[1] - 2) * self.blockSize)

        self.boardPos.append(boardLbPos)
        self.boardPos.append(boardRtPos)

        backgroundVerticesData = [
            0.0, 0.0, -5.0, 1.0, 0.0, 0.0, 0.5,
            self.boardPos[0][0], 0.0, -5.0, 1.0, 0.0, 0.0, 0.5,
            self.boardPos[0][0], self.boardPos[1][1], -5.0, 1.0, 0.0, 0.0, 0.5,
            0.0, self.boardPos[1][1], -5.0, 1.0, 0.0, 0.0, 0.5,

            self.boardPos[1][0], 0.0, -5.0, 1.0, 0.0, 0.0, 0.5,
            self.displaySize[0], 0.0, -5.0, 1.0, 0.0, 0.0, 0.5,
            self.displaySize[0], self.boardPos[1][1], -5.0, 1.0, 0.0, 0.0, 0.5,
            self.boardPos[1][0], self.boardPos[1][1], -5.0, 1.0, 0.0, 0.0, 0.5,

            0.0, 0.0, -5.0, 1.0, 0.0, 0.0, 0.5,
            self.displaySize[0], 0.0, -5.0, 1.0, 0.0, 0.0, 0.5,
            self.displaySize[0], self.boardPos[0][1], -5.0, 1.0, 0.0, 0.0, 0.5,
            0.0, self.boardPos[0][1], -5.0, 1.0, 0.0, 0.0, 0.5
            ]

        backgroundIndicesData = [
            0, 1, 2, 2, 3, 0,
            4, 5, 6, 6, 7, 4,
            8, 9, 10, 10, 11, 8
            ]

        self.backgroundVertices = np.array(backgroundVerticesData, dtype = np.float32)
        self.backgroundIndices = np.array(backgroundIndicesData, dtype = np.uint32)

        backgroundSubAreaVerticesData = [
            25.0, 400.0, -4.5, 0.0, 0.0, 0.0, 1.0,
            175.0, 400.0, -4.5, 0.0, 0.0, 0.0, 1.0,
            175.0, 500.0, -4.5, 0.0, 0.0, 0.0, 1.0,
            25.0, 500.0, -4.5, 0.0, 0.0, 0.0, 1.0,

            475.0, 100.0, -4.5, 0.0, 0.0, 0.0, 1.0,
            775.0, 100.0, -4.5, 0.0, 0.0, 0.0, 1.0,
            775.0, 250.0, -4.5, 0.0, 0.0, 0.0, 1.0,
            475.0, 250.0, -4.5, 0.0, 0.0, 0.0, 1.0
            ]

        backgroundSubAreaIndicesData = [
            0, 1, 2, 2, 3, 0,
            4, 5, 6, 6, 7, 4
            ]

        self.backgroundSubAreaVertices = np.array(backgroundSubAreaVerticesData, dtype = np.float32)
        self.backgroundSubAreaIndices = np.array(backgroundSubAreaIndicesData, dtype = np.uint32)

        backgroundLineVerticesData = [
            0.0, self.boardPos[0][1], -4.0, 1.0, 1.0, 1.0, 1.0, 
            self.displaySize[0], self.boardPos[0][1], -4.0, 1.0, 1.0, 1.0, 1.0,

            0.0, self.boardPos[1][1], -4.0, 1.0, 1.0, 1.0, 1.0,
            self.displaySize[0], self.boardPos[1][1], -4.0, 1.0, 1.0, 1.0, 1.0,

            self.boardPos[0][0], 0.0, -4.0, 1.0, 1.0, 1.0, 1.0,
            self.boardPos[0][0], self.displaySize[1], -4.0, 1.0, 1.0, 1.0, 1.0,

            self.boardPos[1][0], 0.0, -4.0, 1.0, 1.0, 1.0, 1.0,
            self.boardPos[1][0], self.displaySize[1], -4.0, 1.0, 1.0, 1.0, 1.0
            ]

        backgroundLineIndicesData = [
            0, 1,
            2, 3,
            4, 5,
            6, 7
            ]

        self.backgroundLineVertices = np.array(backgroundLineVerticesData, dtype = np.float32)
        self.backgroundLineIndices = np.array(backgroundLineIndicesData, dtype = np.uint32)

        backgroundSubAreaLineVerticesData = [
            25.0, 400.0, -4.0, 1.0, 1.0, 1.0, 1.0,
            175.0, 400.0, -4.0, 1.0, 1.0, 1.0, 1.0,
            175.0, 500.0, -4.0, 1.0, 1.0, 1.0, 1.0,
            25.0, 500.0, -4.0, 1.0, 1.0, 1.0, 1.0,

            475.0, 100.0, -4.0, 1.0, 1.0, 1.0, 1.0,
            775.0, 100.0, -4.0, 1.0, 1.0, 1.0, 1.0,
            775.0, 250.0, -4.0, 1.0, 1.0, 1.0, 1.0,
            475.0, 250.0, -4.0, 1.0, 1.0, 1.0, 1.0
            ]

        backgroundSubAreaLineIndicesData = [
            0, 1, 1, 2, 2, 3, 3, 0,
            4, 5, 5, 6, 6, 7, 7, 4
            ]

        self.backgroundSubAreaLineVertices = np.array(backgroundSubAreaLineVerticesData, dtype = np.float32)
        self.backgroundSubAreaLineIndices = np.array(backgroundSubAreaLineIndicesData, dtype = np.uint32)

        backgroundBoardLineVerticesData = []
        backgroundBoardLineIndicesData = []

        boardLineIndex = 0

        for i in range(self.boardSize[1] - 3):
            lineY = self.boardPos[0][1] + (self.blockSize * (i + 1))

            backgroundBoardLineVerticesData.append(self.boardPos[0][0])
            backgroundBoardLineVerticesData.append(lineY)
            backgroundBoardLineVerticesData.append(-4.0)

            backgroundBoardLineVerticesData.append(1.0)
            backgroundBoardLineVerticesData.append(1.0)
            backgroundBoardLineVerticesData.append(1.0)
            backgroundBoardLineVerticesData.append(0.5)

            backgroundBoardLineVerticesData.append(self.boardPos[1][0])
            backgroundBoardLineVerticesData.append(lineY)
            backgroundBoardLineVerticesData.append(-4.0)

            backgroundBoardLineVerticesData.append(1.0)
            backgroundBoardLineVerticesData.append(1.0)
            backgroundBoardLineVerticesData.append(1.0)
            backgroundBoardLineVerticesData.append(0.5)

            backgroundBoardLineIndicesData.append(2 * boardLineIndex + 0)
            backgroundBoardLineIndicesData.append(2 * boardLineIndex + 1)

            boardLineIndex += 1

        for i in range(self.boardSize[0] - 1):
            lineX = self.boardPos[0][0] + (self.blockSize * (i + 1))

            backgroundBoardLineVerticesData.append(lineX)
            backgroundBoardLineVerticesData.append(self.boardPos[0][1])
            backgroundBoardLineVerticesData.append(-4.0)

            backgroundBoardLineVerticesData.append(1.0)
            backgroundBoardLineVerticesData.append(1.0)
            backgroundBoardLineVerticesData.append(1.0)
            backgroundBoardLineVerticesData.append(0.5)

            backgroundBoardLineVerticesData.append(lineX)
            backgroundBoardLineVerticesData.append(self.boardPos[1][1])
            backgroundBoardLineVerticesData.append(-4.0)

            backgroundBoardLineVerticesData.append(1.0)
            backgroundBoardLineVerticesData.append(1.0)
            backgroundBoardLineVerticesData.append(1.0)
            backgroundBoardLineVerticesData.append(0.5)

            backgroundBoardLineIndicesData.append(2 * boardLineIndex + 0)
            backgroundBoardLineIndicesData.append(2 * boardLineIndex + 1)

            boardLineIndex += 1

        self.backgroundBoardLineVertices = np.array(backgroundBoardLineVerticesData, dtype = np.float32)
        self.backgroundBoardLineIndices = np.array(backgroundBoardLineIndicesData, dtype = np.uint32)

        self.backgroundStuffVerticesList.append(self.backgroundVertices)
        self.backgroundStuffVerticesList.append(self.backgroundSubAreaVertices)
        self.backgroundStuffVerticesList.append(self.backgroundLineVertices)
        self.backgroundStuffVerticesList.append(self.backgroundSubAreaLineVertices)
        self.backgroundStuffVerticesList.append(self.backgroundBoardLineVertices)

        self.backgroundStuffIndicesList.append(self.backgroundIndices)
        self.backgroundStuffIndicesList.append(self.backgroundSubAreaIndices)
        self.backgroundStuffIndicesList.append(self.backgroundLineIndices)
        self.backgroundStuffIndicesList.append(self.backgroundSubAreaLineIndices)
        self.backgroundStuffIndicesList.append(self.backgroundBoardLineIndices)

        for i in range(self.numBackgroundStuff):
            glBindVertexArray(self.backgroundVAO[i])

            glBindBuffer(GL_ARRAY_BUFFER, self.backgroundVBO[i])
            glBufferData(GL_ARRAY_BUFFER, self.backgroundStuffVerticesList[i].nbytes, self.backgroundStuffVerticesList[i], GL_STATIC_DRAW)

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.backgroundEBO[i])
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.backgroundStuffIndicesList[i].nbytes, self.backgroundStuffIndicesList[i], GL_STATIC_DRAW)

            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.backgroundStuffVerticesList[i].itemsize * 7, ctypes.c_void_p(0))

            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, self.backgroundStuffVerticesList[i].itemsize * 7, ctypes.c_void_p(self.backgroundStuffVerticesList[i].itemsize * 3))

        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glBindVertexArray(0)

    def _InitializeShape(self):
        self.allShapes.clear()
        self.allShapeColors.clear()

        self.allShapes.append([0, 1, 2, 3])
        self.allShapeColors.append([1.0, 0.0, 0.0])
        self.allShapes.append([0, 1, 2, 5])
        self.allShapeColors.append([0.0, 0.76, 0.0])
        self.allShapes.append([0, 1, 4, 5])
        self.allShapeColors.append([0.15, 0.18, 0.85])
        self.allShapes.append([0, 1, 2, 6])
        self.allShapeColors.append([0.9, 0.73, 0.0])
        self.allShapes.append([0, 1, 2, 4])
        self.allShapeColors.append([0.95, 0.0, 0.89])
        self.allShapes.append([1, 5, 2, 4])
        self.allShapeColors.append([0.0, 0.9, 0.91])
        self.allShapes.append([0, 5, 1, 6])
        self.allShapeColors.append([1.0, 0.56, 0.0])

        allBlocksVerticesData = []
        allBlocksIndicesData = []

        for r in range(self.boardSize[1] - 2):
            for c in range(self.boardSize[0]):
                for v in range(self.numVerticesOneBlock):
                    allBlocksVerticesData.append(0.0)
                    allBlocksVerticesData.append(0.0)
                    allBlocksVerticesData.append(0.0)

                    allBlocksVerticesData.append(1.0)
                    allBlocksVerticesData.append(1.0)
                    allBlocksVerticesData.append(1.0)
                    allBlocksVerticesData.append(1.0)

        for r in range(self.boardSize[1] - 2):
            rOffset = r * self.boardSize[0] * self.numVerticesOneBlock

            for c in range(self.boardSize[0]):
                cOffset = c * self.numVerticesOneBlock

                allBlocksIndicesData.append(rOffset + cOffset + 0)
                allBlocksIndicesData.append(rOffset + cOffset + 1)
                allBlocksIndicesData.append(rOffset + cOffset + 2)

                allBlocksIndicesData.append(rOffset + cOffset + 2)
                allBlocksIndicesData.append(rOffset + cOffset + 3)
                allBlocksIndicesData.append(rOffset + cOffset + 0)

        self.allBlocksVertices = np.array(allBlocksVerticesData, dtype = np.float32)
        self.allBlocksIndices = np.array(allBlocksIndicesData, dtype = np.uint32)

        preShapeVerticesData = []
        preShapeIndicesData = []

        curShapeVerticesData = []
        curShapeIndicesData = []

        for i in range(self.numBlocksOneShape):
            for j in range(self.numVerticesOneBlock):
                preShapeVerticesData.append(0.0)
                preShapeVerticesData.append(0.0)
                preShapeVerticesData.append(0.0)

                preShapeVerticesData.append(1.0)
                preShapeVerticesData.append(1.0)
                preShapeVerticesData.append(1.0)
                preShapeVerticesData.append(1.0)

                curShapeVerticesData.append(0.0)
                curShapeVerticesData.append(0.0)
                curShapeVerticesData.append(0.0)

                curShapeVerticesData.append(1.0)
                curShapeVerticesData.append(1.0)
                curShapeVerticesData.append(1.0)
                curShapeVerticesData.append(1.0)

        for i in range(self.numBlocksOneShape):
            iOffset = i * self.numVerticesOneBlock

            preShapeIndicesData.append(iOffset + 0)
            preShapeIndicesData.append(iOffset + 1)
            preShapeIndicesData.append(iOffset + 2)

            preShapeIndicesData.append(iOffset + 2)
            preShapeIndicesData.append(iOffset + 3)
            preShapeIndicesData.append(iOffset + 0)

            curShapeIndicesData.append(iOffset + 0)
            curShapeIndicesData.append(iOffset + 1)
            curShapeIndicesData.append(iOffset + 2)

            curShapeIndicesData.append(iOffset + 2)
            curShapeIndicesData.append(iOffset + 3)
            curShapeIndicesData.append(iOffset + 0)

        self.preShapeVertices = np.array(preShapeVerticesData, dtype = np.float32)
        self.preShapeIndices = np.array(preShapeIndicesData, dtype = np.uint32)

        self.curShapeVertices = np.array(curShapeVerticesData, dtype = np.float32)
        self.curShapeIndices = np.array(curShapeIndicesData, dtype = np.uint32)

        shapeDistributionVerticesData = []
        shapeDistributionIndicesData = []

        for i in range(self.numShapes):
            for j in range(self.numVerticesOneBlock):
                shapeDistributionVerticesData.append(0.0)
                shapeDistributionVerticesData.append(0.0)
                shapeDistributionVerticesData.append(0.0)

                shapeColor = self.allShapeColors[i]

                shapeDistributionVerticesData.append(shapeColor[0])
                shapeDistributionVerticesData.append(shapeColor[1])
                shapeDistributionVerticesData.append(shapeColor[2])
                shapeDistributionVerticesData.append(0.8)

        for i in range(self.numShapes):
            iOffset = i * self.numBlocksOneShape

            shapeDistributionIndicesData.append(iOffset + 0)
            shapeDistributionIndicesData.append(iOffset + 1)
            shapeDistributionIndicesData.append(iOffset + 2)

            shapeDistributionIndicesData.append(iOffset + 2)
            shapeDistributionIndicesData.append(iOffset + 3)
            shapeDistributionIndicesData.append(iOffset + 0)

        self.shapeDistributionVertices = np.array(shapeDistributionVerticesData, dtype = np.float32)
        self.shapeDistributionIndices = np.array(shapeDistributionIndicesData, dtype = np.uint32)

        self.gameStuffVerticesList.clear()
        self.gameStuffIndicesList.clear()

        self.gameStuffVerticesList.append(self.allBlocksVertices)
        self.gameStuffVerticesList.append(self.preShapeVertices)
        self.gameStuffVerticesList.append(self.curShapeVertices)
        self.gameStuffVerticesList.append(self.shapeDistributionVertices)

        self.gameStuffIndicesList.append(self.allBlocksIndices)
        self.gameStuffIndicesList.append(self.preShapeIndices)
        self.gameStuffIndicesList.append(self.curShapeIndices)
        self.gameStuffIndicesList.append(self.shapeDistributionIndices)

        for i in range(self.numGameStuff):
            glBindVertexArray(self.gameVAO[i])

            glBindBuffer(GL_ARRAY_BUFFER, self.gameVBO[i])
            glBufferData(GL_ARRAY_BUFFER, self.gameStuffVerticesList[i].nbytes, self.gameStuffVerticesList[i], GL_DYNAMIC_DRAW)

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.gameEBO[i])
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.gameStuffIndicesList[i].nbytes, self.gameStuffIndicesList[i], GL_STATIC_DRAW)

            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.gameStuffVerticesList[i].itemsize * self.numVertexComponents, ctypes.c_void_p(0))

            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, self.gameStuffVerticesList[i].itemsize * self.numVertexComponents, ctypes.c_void_p(self.gameStuffVerticesList[i].itemsize * 3))

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        
        glBindVertexArray(0)

        self.allShapeDistribution = [0 for i in range(self.numShapes)]

        self.preShapeIdx = random.randrange(0, self.numShapes)

        self._CreateNewShape()

    def _CheckNewPosition(self):
        for i in range(self.numBlocksOneShape):
            if self.curShape[i][0] < 1 or self.boardSize[0] < self.curShape[i][0]:
                return False
            elif self.curShape[i][1] < 1 or self.boardSize[1] < self.curShape[i][1]:
                return False
            elif self.allBlocks[self.curShape[i][1] - 1][self.curShape[i][0] - 1] != 255:
                return False

        return True

    def _ClearLine(self):
        checkLineIdx = 0

        for r in range(self.boardSize[1] - 1):
            blocksCntOneLine = 0

            for c in range(self.boardSize[0]):
                if self.allBlocks[r][c] != 255:
                    blocksCntOneLine += 1

                self.allBlocks[checkLineIdx][c] = self.allBlocks[r][c]

            if blocksCntOneLine != self.boardSize[0]:
                checkLineIdx += 1

    def _UpdateShape(self):
        tmpCurShape = []

        for i in range(self.numBlocksOneShape):
            tmpCurShape.append([self.curShape[i][0], self.curShape[i][1]])

        if self.move[0] == True and self.stackShapeImmediately == False:
            self.move[0] = False

            for i in range(self.numBlocksOneShape):
                self.curShape[i][0] -= 1

            if self._CheckNewPosition() == False:
                for i in range(self.numBlocksOneShape):
                    self.curShape[i][0] = tmpCurShape[i][0]

        elif self.move[1] == True and self.stackShapeImmediately == False:
            self.move[1] = False

            for i in range(self.numBlocksOneShape):
                self.curShape[i][0] += 1

            if self._CheckNewPosition() == False:
                for i in range(self.numBlocksOneShape):
                    self.curShape[i][0] = tmpCurShape[i][0]

        elif self.move[2] == True or self.stackShapeImmediately == True:
            self.move[2] = False

            for i in range(self.numBlocksOneShape):
                self.curShape[i][1] -= 1

            if self._CheckNewPosition() == False:
                for i in range(self.numBlocksOneShape):
                    self.curShape[i][1] = tmpCurShape[i][1]

                    if self.curShape[i][1] > (self.boardSize[1] - 2):
                        self.Restart()
                        return

                    self.allBlocks[self.curShape[i][1] - 1][self.curShape[i][0] - 1] = self.curShapeIdx

                self.stackShapeImmediately = False

                self.allShapeDistribution[self.curShapeIdx] += 1

                self._ClearLine()

                self._CreateNewShape()

        elif self.rotate == True and self.curShapeIdx != 2 and self.stackShapeImmediately == False:
            self.rotate = False

            rotateCenterIdx = 1

            for i in range(self.numBlocksOneShape):
                if i == rotateCenterIdx:
                    continue

                deltaX = self.curShape[i][0] - self.curShape[rotateCenterIdx][0]
                deltaY = self.curShape[i][1] - self.curShape[rotateCenterIdx][1]

                rotatedDeltaX = -deltaY
                rotatedDeltaY = deltaX

                self.curShape[i][0] = self.curShape[rotateCenterIdx][0] + rotatedDeltaX
                self.curShape[i][1] = self.curShape[rotateCenterIdx][1] + rotatedDeltaY

            if self._CheckNewPosition() == False:
                for i in range(self.numBlocksOneShape):
                    self.curShape[i][0] = tmpCurShape[i][0]
                    self.curShape[i][1] = tmpCurShape[i][1]

    def _DrawBackground(self):
        glPushAttrib(GL_COLOR_BUFFER_BIT | GL_ENABLE_BIT | GL_LINE_BIT)

        glEnable(GL_BLEND)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        backgroundStuffIndex = 0

        glBindVertexArray(self.backgroundVAO[backgroundStuffIndex])
        glDrawElements(GL_TRIANGLES, len(self.backgroundStuffIndicesList[backgroundStuffIndex]), GL_UNSIGNED_INT, None)

        backgroundStuffIndex += 1

        glBindVertexArray(self.backgroundVAO[backgroundStuffIndex])
        glDrawElements(GL_TRIANGLES, len(self.backgroundStuffIndicesList[backgroundStuffIndex]), GL_UNSIGNED_INT, None)

        backgroundStuffIndex += 1

        glLineWidth(2.0)

        glBindVertexArray(self.backgroundVAO[backgroundStuffIndex])
        glDrawElements(GL_LINES, len(self.backgroundStuffIndicesList[backgroundStuffIndex]), GL_UNSIGNED_INT, None)

        backgroundStuffIndex += 1

        glBindVertexArray(self.backgroundVAO[backgroundStuffIndex])
        glDrawElements(GL_LINES, len(self.backgroundStuffIndicesList[backgroundStuffIndex]), GL_UNSIGNED_INT, None)

        backgroundStuffIndex += 1

        glLineWidth(1.0)

        glBindVertexArray(self.backgroundVAO[backgroundStuffIndex])
        glDrawElements(GL_LINES, len(self.backgroundStuffIndicesList[backgroundStuffIndex]), GL_UNSIGNED_INT, None)

        glBindVertexArray(0)

        glPopAttrib()

    def _DrawStackBlocks(self):
        glPushAttrib(GL_COLOR_BUFFER_BIT | GL_ENABLE_BIT)

        glEnable(GL_BLEND)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        for i in range(len(self.allBlocksVertices)):
            self.allBlocksVertices[i] = 0.0

        for r in range(self.boardSize[1] - 2):
            rOffset = r * self.boardSize[0] * self.numVerticesOneBlock * self.numVertexComponents

            for c in range(self.boardSize[0]):
                if self.allBlocks[r][c] != 255:
                    cOffset = c * self.numVerticesOneBlock * self.numVertexComponents

                    posX = self.boardPos[0][0] + (c + 1) * self.blockSize
                    posY = self.boardPos[0][1] + (r + 1) * self.blockSize

                    blockColor = self.allShapeColors[self.allBlocks[r][c]]

                    verticesIndex = 0
                    vOffset = verticesIndex * self.numVertexComponents

                    self.allBlocksVertices[rOffset + cOffset + vOffset + 0] = posX - self.blockSize
                    self.allBlocksVertices[rOffset + cOffset + vOffset + 1] = posY - self.blockSize
                    self.allBlocksVertices[rOffset + cOffset + vOffset + 2] = 0.0

                    self.allBlocksVertices[rOffset + cOffset + vOffset + 3] = blockColor[0]
                    self.allBlocksVertices[rOffset + cOffset + vOffset + 4] = blockColor[1]
                    self.allBlocksVertices[rOffset + cOffset + vOffset + 5] = blockColor[2]
                    self.allBlocksVertices[rOffset + cOffset + vOffset + 6] = 1.0

                    verticesIndex += 1
                    vOffset = verticesIndex * self.numVertexComponents

                    self.allBlocksVertices[rOffset + cOffset + vOffset + 0] = posX
                    self.allBlocksVertices[rOffset + cOffset + vOffset + 1] = posY - self.blockSize
                    self.allBlocksVertices[rOffset + cOffset + vOffset + 2] = 0.0

                    self.allBlocksVertices[rOffset + cOffset + vOffset + 3] = blockColor[0]
                    self.allBlocksVertices[rOffset + cOffset + vOffset + 4] = blockColor[1]
                    self.allBlocksVertices[rOffset + cOffset + vOffset + 5] = blockColor[2]
                    self.allBlocksVertices[rOffset + cOffset + vOffset + 6] = 1.0

                    verticesIndex += 1
                    vOffset = verticesIndex * self.numVertexComponents

                    self.allBlocksVertices[rOffset + cOffset + vOffset + 0] = posX
                    self.allBlocksVertices[rOffset + cOffset + vOffset + 1] = posY
                    self.allBlocksVertices[rOffset + cOffset + vOffset + 2] = 0.0

                    self.allBlocksVertices[rOffset + cOffset + vOffset + 3] = blockColor[0]
                    self.allBlocksVertices[rOffset + cOffset + vOffset + 4] = blockColor[1]
                    self.allBlocksVertices[rOffset + cOffset + vOffset + 5] = blockColor[2]
                    self.allBlocksVertices[rOffset + cOffset + vOffset + 6] = 1.0

                    verticesIndex += 1
                    vOffset = verticesIndex * self.numVertexComponents

                    self.allBlocksVertices[rOffset + cOffset + vOffset + 0] = posX - self.blockSize
                    self.allBlocksVertices[rOffset + cOffset + vOffset + 1] = posY
                    self.allBlocksVertices[rOffset + cOffset + vOffset + 2] = 0.0

                    self.allBlocksVertices[rOffset + cOffset + vOffset + 3] = blockColor[0]
                    self.allBlocksVertices[rOffset + cOffset + vOffset + 4] = blockColor[1]
                    self.allBlocksVertices[rOffset + cOffset + vOffset + 5] = blockColor[2]
                    self.allBlocksVertices[rOffset + cOffset + vOffset + 6] = 1.0

        glBindBuffer(GL_ARRAY_BUFFER, self.gameVBO[0])
        glBufferSubData(GL_ARRAY_BUFFER, 0, self.gameStuffVerticesList[0].nbytes, self.gameStuffVerticesList[0])
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glBindVertexArray(self.gameVAO[0])
        glDrawElements(GL_TRIANGLES, len(self.gameStuffIndicesList[0]), GL_UNSIGNED_INT, None)

        glBindVertexArray(0)

        glPopAttrib()

    def _DrawShape(self):
        glPushAttrib(GL_COLOR_BUFFER_BIT | GL_ENABLE_BIT)

        glEnable(GL_BLEND)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        for i in range (self.numBlocksOneShape):
            iOffset = i * self.numVerticesOneBlock * self.numVertexComponents

            posX = self.boardPos[0][0] + self.curShape[i][0] * self.blockSize
            posY = self.boardPos[0][1] + self.curShape[i][1] * self.blockSize

            verticesIndex = 0
            vOffset = verticesIndex * self.numVertexComponents

            self.curShapeVertices[iOffset + vOffset + 0] = posX - self.blockSize
            self.curShapeVertices[iOffset + vOffset + 1] = posY - self.blockSize
            self.curShapeVertices[iOffset + vOffset + 2] = 0.0

            verticesIndex += 1
            vOffset = verticesIndex * self.numVertexComponents

            self.curShapeVertices[iOffset + vOffset + 0] = posX
            self.curShapeVertices[iOffset + vOffset + 1] = posY - self.blockSize
            self.curShapeVertices[iOffset + vOffset + 2] = 0.0

            verticesIndex += 1
            vOffset = verticesIndex * self.numVertexComponents

            self.curShapeVertices[iOffset + vOffset + 0] = posX
            self.curShapeVertices[iOffset + vOffset + 1] = posY
            self.curShapeVertices[iOffset + vOffset + 2] = 0.0

            verticesIndex += 1
            vOffset = verticesIndex * self.numVertexComponents

            self.curShapeVertices[iOffset + vOffset + 0] = posX - self.blockSize
            self.curShapeVertices[iOffset + vOffset + 1] = posY
            self.curShapeVertices[iOffset + vOffset + 2] = 0.0

        glBindBuffer(GL_ARRAY_BUFFER, self.gameVBO[1])
        glBufferSubData(GL_ARRAY_BUFFER, 0, self.gameStuffVerticesList[1].nbytes, self.gameStuffVerticesList[1])
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glBindVertexArray(self.gameVAO[1])
        glDrawElements(GL_TRIANGLES, len(self.gameStuffIndicesList[1]), GL_UNSIGNED_INT, None)

        glBindVertexArray(0)

        glPopAttrib()

    def _DrawPreShape(self):
        glPushAttrib(GL_COLOR_BUFFER_BIT | GL_ENABLE_BIT)

        glEnable(GL_BLEND)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        for i in range(self.numBlocksOneShape):
            iOffset = i * self.numVerticesOneBlock * self.numVertexComponents

            posX = (self.backgroundSubAreaVertices[0] + self.backgroundSubAreaVertices[7]) / 2.0 - self.blockSize
            posY = (self.backgroundSubAreaVertices[8] + self.backgroundSubAreaVertices[15]) / 2.0 + 20.0

            posX += self.preShape[i][0] * self.blockSize
            posY -= self.preShape[i][1] * self.blockSize

            verticesIndex = 0
            vOffset = verticesIndex * self.numVertexComponents

            self.preShapeVertices[iOffset + vOffset + 0] = posX - self.blockSize
            self.preShapeVertices[iOffset + vOffset + 1] = posY - self.blockSize
            self.preShapeVertices[iOffset + vOffset + 2] = 0.0

            verticesIndex += 1
            vOffset = verticesIndex * self.numVertexComponents

            self.preShapeVertices[iOffset + vOffset + 0] = posX
            self.preShapeVertices[iOffset + vOffset + 1] = posY - self.blockSize
            self.preShapeVertices[iOffset + vOffset + 2] = 0.0

            verticesIndex += 1
            vOffset = verticesIndex * self.numVertexComponents

            self.preShapeVertices[iOffset + vOffset + 0] = posX
            self.preShapeVertices[iOffset + vOffset + 1] = posY
            self.preShapeVertices[iOffset + vOffset + 2] = 0.0

            verticesIndex += 1
            vOffset = verticesIndex * self.numVertexComponents

            self.preShapeVertices[iOffset + vOffset + 0] = posX - self.blockSize
            self.preShapeVertices[iOffset + vOffset + 1] = posY
            self.preShapeVertices[iOffset + vOffset + 2] = 0.0

        glBindBuffer(GL_ARRAY_BUFFER, self.gameVBO[2])
        glBufferSubData(GL_ARRAY_BUFFER, 0, self.gameStuffVerticesList[2].nbytes, self.gameStuffVerticesList[2])
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glBindVertexArray(self.gameVAO[2])
        glDrawElements(GL_TRIANGLES, len(self.gameStuffIndicesList[2]), GL_UNSIGNED_INT, None)

        glBindVertexArray(0)

        glPopAttrib()

    def _DrawShapeDistribution(self):
        glPushAttrib(GL_COLOR_BUFFER_BIT | GL_ENABLE_BIT)

        glEnable(GL_BLEND)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        posX = (self.backgroundSubAreaVertices[35] - self.backgroundSubAreaVertices[28]) - (self.numShapes * self.blockSize)
        posX = self.backgroundSubAreaVertices[28] + posX / 2.0
        posY = self.backgroundSubAreaVertices[29] + 20.0

        for i in range(self.numShapes):
            iOffset = i * self.numVerticesOneBlock * self.numVertexComponents

            posX += self.blockSize

            verticesIndex = 0
            vOffset = verticesIndex * self.numVertexComponents

            self.shapeDistributionVertices[iOffset + vOffset + 0] = posX - self.blockSize
            self.shapeDistributionVertices[iOffset + vOffset + 1] = posY
            self.shapeDistributionVertices[iOffset + vOffset + 2] = 0.0

            verticesIndex += 1
            vOffset = verticesIndex * self.numVertexComponents

            self.shapeDistributionVertices[iOffset + vOffset + 0] = posX
            self.shapeDistributionVertices[iOffset + vOffset + 1] = posY
            self.shapeDistributionVertices[iOffset + vOffset + 2] = 0.0

            verticesIndex += 1
            vOffset = verticesIndex * self.numVertexComponents

            self.shapeDistributionVertices[iOffset + vOffset + 0] = posX
            self.shapeDistributionVertices[iOffset + vOffset + 1] = posY + self.allShapeDistribution[i]
            self.shapeDistributionVertices[iOffset + vOffset + 2] = 0.0

            verticesIndex += 1
            vOffset = verticesIndex * self.numVertexComponents

            self.shapeDistributionVertices[iOffset + vOffset + 0] = posX - self.blockSize
            self.shapeDistributionVertices[iOffset + vOffset + 1] = posY + self.allShapeDistribution[i]
            self.shapeDistributionVertices[iOffset + vOffset + 2] = 0.0

        glBindBuffer(GL_ARRAY_BUFFER, self.gameVBO[3])
        glBufferSubData(GL_ARRAY_BUFFER, 0, self.gameStuffVerticesList[3].nbytes, self.gameStuffVerticesList[3])
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glBindVertexArray(self.gameVAO[3])
        glDrawElements(GL_TRIANGLES, len(self.gameStuffIndicesList[3]), GL_UNSIGNED_INT, None)

        glBindVertexArray(0)

        glPopAttrib()

gSceneManager = SceneManager(False)
gInputManager = InputManager()


def HandleWindowSizeCallback(glfwWindow, width, height):
    glViewport(0, 0, width, height)

    gSceneManager.SetDisplaySize(width, height)

def HandleKeyCallback(glfwWindow, key, scanCode, action, modes):
    if action == glfw.PRESS:
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(glfwWindow, glfw.TRUE)

        if key == glfw.KEY_SPACE:
            gInputManager.SetKeyState(glfw.KEY_SPACE, True)

        if key == glfw.KEY_1:
            gInputManager.SetKeyState('1', True)
        if key == glfw.KEY_2:
            gInputManager.SetKeyState('2', True)

        if key == glfw.KEY_B:
            gInputManager.SetKeyState('B', True)        
        if key == glfw.KEY_I:
            gInputManager.SetKeyState('I', True) 
        if key == glfw.KEY_P:
            gInputManager.SetKeyState('P', True)            
        if key == glfw.KEY_R:
            gInputManager.SetKeyState('R', True)

        if key == glfw.KEY_W:
            gInputManager.SetKeyState('W', True)
        elif key == glfw.KEY_S:
            gInputManager.SetKeyState('S', True)
        elif key == glfw.KEY_A:
            gInputManager.SetKeyState('A', True)
        elif key == glfw.KEY_D:
            gInputManager.SetKeyState('D', True)

        if key == glfw.KEY_UP:
            gInputManager.SetKeyState(glfw.KEY_UP, True)
        elif key == glfw.KEY_DOWN:
            gInputManager.SetKeyState(glfw.KEY_DOWN, True)
        elif key == glfw.KEY_LEFT:
            gInputManager.SetKeyState(glfw.KEY_LEFT, True)
        elif key == glfw.KEY_RIGHT:
            gInputManager.SetKeyState(glfw.KEY_RIGHT, True)

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
    if gSceneManager.GetPause() == True:
        return

    if gInputManager.GetMouseEntered() == False:
        gInputManager.SetLastMousePos([xPos, yPos])
        gInputManager.SetMouseEntered(True)

    lastPos = gInputManager.GetLastMousePos()
    xOffset = lastPos[0] - xPos
    yOffset = lastPos[1] - yPos

    gInputManager.SetLastMousePos([xPos, yPos])

    camera = gSceneManager.GetCamera()

    if gSceneManager.GetView3D() == True:
        camera.ProcessMouseMovement(xOffset, yOffset)

    displaySize = gSceneManager.GetDisplaySize()

    mouseCheckInterval = 20

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

    glfwWindow = glfw.create_window(displaySize[0], displaySize[1], "Tetris.Part 1", None, None)

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

    shader = Shader(vertexShaderCode, fragmentShaderCode)

    gSceneManager.InitializeOpenGL()
    gSceneManager.SetCamera(Camera())
    gSceneManager.MakeFont()
    gSceneManager.AddObject(Tetris(displaySize))

    prjMat = []
    viewMat = []

    lastElapsedTime = glfw.get_time()
    deltaTime = 0.0

    while glfw.window_should_close(glfwWindow) == False:
        glfw.poll_events()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        gSceneManager.Update(deltaTime)

        prjMat = gSceneManager.GetOrthoPrjMat()
        viewMat = gSceneManager.GetViewMat()

        shader.Use()

        shader.SetMat4('prjMat', prjMat)
        shader.SetMat4('viewMat', viewMat)

        gSceneManager.DrawObjects()
        
        glUseProgram(0)

        gSceneManager.DrawProgramInfo(deltaTime)        

        glfw.swap_buffers(glfwWindow)

        deltaTime = glfw.get_time() - lastElapsedTime
        lastElapsedTime = glfw.get_time()


glfw.terminate()


if __name__ == "__main__":
    Main()  