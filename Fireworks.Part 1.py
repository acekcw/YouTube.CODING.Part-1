
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

    gl_Position = prjMat * viewMat *  vec4(aPos, 1.0);
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
    def __init__(self):
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

        self.objects = []
        self.maxNumFireworks = 1000
        self.fireworksCount = 0
        self.font = None

        self.deltaTime = 0.0
        self.dirty = True

        self.programInfo = True
        self.numProgramInfoElement = 6

        self.trail = False

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
        if object.__class__.__name__ == "Firework":
            if random.random() < 0.5:
                if self.fireworksCount < self.maxNumFireworks:
                    self.fireworksCount += 1

                    object.SetTrail(self.trail)
                else:
                    return
            else:
                return

        self.objects.append(object)

    def UpdateAboutInput(self):
        numObjects = len(self.objects)

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
            for i in range(numObjects - 1, -1, -1):
                if self.objects[i].__class__.__name__ == "Firework":
                    self.objects.pop(i)
                    self.fireworksCount -= 1
                else:
                    self.objects[i].Restart()
            gInputManager.SetKeyState('R', False)
        if gInputManager.GetKeyState('T') == True:
            self.trail = not self.trail
            gInputManager.SetKeyState('T', False)

    def Update(self, deltaTime):
        self.UpdateAboutInput()

        if self.pause == True:
            return

        numObjects = len(self.objects)

        for i in range(numObjects - 1, -1, -1):
            self.objects[i].Update(deltaTime)

            if self.objects[i].__class__.__name__ == "Firework":
                if self.objects[i].Done():
                    self.objects.pop(i)
                    self.fireworksCount -= 1

        self.SetCameraPos()

        self.SailCamera()

        if self.dirty == False:
            return

        self.perspectivePrjMat = glm.perspective(self.fovy, self.aspect, self.near, self.far)

        self.orthoPrjMat = glm.ortho(0, self.displaySize[0], 0, self.displaySize[1], -1.0, 1.0)

        self.viewMat = self.camera.GetViewMat()

        self.deltaTime += deltaTime
        self.dirty = False

    def DrawObjects(self):
        numObjects = len(self.objects)

        #print('numObjects: {0}'.format(numObjects))

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
            14, 15
            ]

        self.programInfoAreaVertices = np.array(programInfoAreaVerticesData, dtype = np.float32)
        self.programInfoAreaIndices = np.array(programInfoAreaIndicesData, dtype = np.uint32)

    def _DrawProgramInfoArea(self):
        glPushMatrix()
        glLoadIdentity()

        glPushAttrib(GL_COLOR_BUFFER_BIT | GL_ENABLE_BIT | GL_LINE_BIT)

        glEnable(GL_BLEND)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

        glVertexPointer(3, GL_FLOAT, self.programInfoAreaVertices.itemsize * 7, ctypes.c_void_p(self.programInfoAreaVertices.ctypes.data))
        glColorPointer(4, GL_FLOAT, self.programInfoAreaVertices.itemsize * 7, ctypes.c_void_p(self.programInfoAreaVertices.ctypes.data + self.programInfoAreaVertices.itemsize * 3))

        glDrawElements(GL_LINES, len(self.programInfoAreaIndices), GL_UNSIGNED_INT, self.programInfoAreaIndices)

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

        glPopAttrib()

        glPopMatrix()
       
    def _DrawProgramInfo(self, deltaTime):        
        glPushMatrix()
        glLoadIdentity()

        glPushAttrib(GL_COLOR_BUFFER_BIT | GL_ENABLE_BIT)

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

        infoText.append('.Add Trail(T): ')
        infoTextIndex += 1

        if self.trail == True:
            infoText[infoTextIndex] += "On"
        else:
            infoText[infoTextIndex] += "Off"

        infoText.append('.# NumFireworks: {0}'.format(self.fireworksCount))
        infoTextIndex += 1

        textPosX = 590.0
        textPosY = 570.0

        upMaxIndex = 4

        for i in range(upMaxIndex):
            glTranslate(textPosX, textPosY, 0.0)
            
            glListBase(self.font.GetListOffset())
            glCallLists([ord(c) for c in infoText[i]])

            glPopMatrix()
            glPushMatrix()
            glLoadIdentity()

            textPosY -= 20

        textPosY -= 85.0

        for i in range(upMaxIndex, self.numProgramInfoElement - 1):
            glTranslate(textPosX, textPosY, 0.0)

            glListBase(self.font.GetListOffset())
            glCallLists([ord(c) for c in infoText[i]])

            glPopMatrix()
            glPushMatrix()
            glLoadIdentity()

            textPosY -= 20

        glTranslate(textPosX, textPosY, 0.0)

        glCallLists([ord(c) for c in infoText[self.numProgramInfoElement - 1]])

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
        self.cameraPos = glm.vec3(0.0, 0.0, 30.0)
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

        self.rotDegree += deltaTime * 50.0

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

class Firework:
    def __init__(self, externalForce):
        self.particlesBeforeExplosionVertices = []
        self.particlesBeforeExplosionTrailVertices = []
        self.particlesAfterExplosionVertices =[]
        self.particlesAfterExplosionTrailVertices = []

        self.externalForce = externalForce
        self.numTrailParticles = 20
        self.trail = False

        posInterval = 30.0

        particleBeforeExplosionPos = glm.vec3(0.0, 0.0, 0.0)
        particleBeforeExplosionPos.x = (random.random() - 0.5) * posInterval
        particleBeforeExplosionPos.y = -10.0
        particleBeforeExplosionPos.z = (random.random() - 0.5) * posInterval

        self.particlesBeforeExplosion = Particle(self.externalForce, particleBeforeExplosionPos, False, self.numTrailParticles)
        self.particlesAfterExplosion = []

        self.color = [random.random(), random.random(), random.random()]

        self.numParticlesAfterExplosion = 100
        self.exploded = False

        self.VAO = glGenVertexArrays(4)
        self.VBO = glGenBuffers(4)

        self._InitializeVertexArray()

    def SetTrail(self, value):
        self.trail = value

    def Update(self, deltaTime):
        if self.exploded == False:
            self.particlesBeforeExplosion.Update(deltaTime)

            particlesBeforeExplosionVel = self.particlesBeforeExplosion.GetVelocity()

            if particlesBeforeExplosionVel.y < 0.0:
                self.exploded = True

                self.Explode()
        else:
            numCurrentParticlesAfterExplosion = len(self.particlesAfterExplosion)

            for i in range(numCurrentParticlesAfterExplosion - 1, -1, -1):
                self.particlesAfterExplosion[i].Update(deltaTime)

                if self.particlesAfterExplosion[i].Done():
                    self.particlesAfterExplosion.pop(i)

    def Explode(self):
        for i in range(self.numParticlesAfterExplosion):
            particlesBeforeExplosionPos = self.particlesBeforeExplosion.GetPosition()
            self.particlesAfterExplosion.append(Particle(self.externalForce, particlesBeforeExplosionPos, True, self.numTrailParticles))

    def Done(self):
        if self.exploded == True and len(self.particlesAfterExplosion) == 0:
            return True
        else:
            return False

    def Draw(self):
        glPushAttrib(GL_COLOR_BUFFER_BIT | GL_ENABLE_BIT | GL_POINT_BIT)

        glEnable(GL_POINT_SMOOTH)
        glEnable(GL_BLEND)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        if self.exploded == False:
            glPointSize(5.0)
            
            particlesBeforeExplosionPos = self.particlesBeforeExplosion.GetPosition()

            self.particlesBeforeExplosionVertices[0] = particlesBeforeExplosionPos.x
            self.particlesBeforeExplosionVertices[1] = particlesBeforeExplosionPos.y
            self.particlesBeforeExplosionVertices[2] = particlesBeforeExplosionPos.z

            glBindBuffer(GL_ARRAY_BUFFER, self.VBO[0])
            glBufferSubData(GL_ARRAY_BUFFER, 0, self.particlesBeforeExplosionVertices.itemsize * 3, self.particlesBeforeExplosionVertices)
            glBindBuffer(GL_ARRAY_BUFFER, 0)

            glBindVertexArray(self.VAO[0])
            glDrawArrays(GL_POINTS, 0, 1)

            if self.trail == True:
                particlesBeforeExplosionTrailPos = self.particlesBeforeExplosion.GetTrailPosition()

                alphaInterval = 1.0 / self.numTrailParticles

                for i in range(self.numTrailParticles):
                    self.particlesBeforeExplosionTrailVertices[i * 7 + 0] = particlesBeforeExplosionTrailPos[i].x
                    self.particlesBeforeExplosionTrailVertices[i * 7 + 1] = particlesBeforeExplosionTrailPos[i].y
                    self.particlesBeforeExplosionTrailVertices[i * 7 + 2] = particlesBeforeExplosionTrailPos[i].z

                    self.particlesBeforeExplosionTrailVertices[i * 7 + 6] = 1.0 - (i * alphaInterval)

                glBindBuffer(GL_ARRAY_BUFFER, self.VBO[1])
                glBufferSubData(GL_ARRAY_BUFFER, 0, self.particlesBeforeExplosionTrailVertices.nbytes, self.particlesBeforeExplosionTrailVertices)
                glBindBuffer(GL_ARRAY_BUFFER, 0)

                glBindVertexArray(self.VAO[1])
                glDrawArrays(GL_POINTS, 0, 1 * self.numTrailParticles)

            glBindVertexArray(0)
        else:
            glPointSize(2.0)

            numCurrentParticlesAfterExplosion = len(self.particlesAfterExplosion)

            for i in range(numCurrentParticlesAfterExplosion):
                particlesAfterExplosionPos = self.particlesAfterExplosion[i].GetPosition()
                particlesAfterExplosionLifespan = self.particlesAfterExplosion[i].GetLifespan()

                self.particlesAfterExplosionVertices[i * 7 + 0] = particlesAfterExplosionPos.x
                self.particlesAfterExplosionVertices[i * 7 + 1] = particlesAfterExplosionPos.y
                self.particlesAfterExplosionVertices[i * 7 + 2] = particlesAfterExplosionPos.z

                self.particlesAfterExplosionVertices[i * 7 + 6] = particlesAfterExplosionLifespan

                if self.trail == True:
                    particlesAfterExplosionTrailPos = self.particlesAfterExplosion[i].GetTrailPosition()

                    alphaInterval = particlesAfterExplosionLifespan / self.numTrailParticles

                    for j in range(self.numTrailParticles):
                        self.particlesAfterExplosionTrailVertices[(i * self.numTrailParticles * 7) + (j * 7) + 0] = particlesAfterExplosionTrailPos[j].x
                        self.particlesAfterExplosionTrailVertices[(i * self.numTrailParticles * 7) + (j * 7) + 1] = particlesAfterExplosionTrailPos[j].y
                        self.particlesAfterExplosionTrailVertices[(i * self.numTrailParticles * 7) + (j * 7) + 2] = particlesAfterExplosionTrailPos[j].z

                        self.particlesAfterExplosionTrailVertices[(i * self.numTrailParticles * 7) + (j * 7) + 6] = particlesAfterExplosionLifespan - (j * alphaInterval)

            glBindBuffer(GL_ARRAY_BUFFER, self.VBO[2])
            glBufferSubData(GL_ARRAY_BUFFER, 0, self.particlesAfterExplosionVertices.nbytes, self.particlesAfterExplosionVertices)
            glBindBuffer(GL_ARRAY_BUFFER, 0)

            glBindVertexArray(self.VAO[2])
            glDrawArrays(GL_POINTS, 0, self.numParticlesAfterExplosion)

            if self.trail == True:
                glBindBuffer(GL_ARRAY_BUFFER, self.VBO[3])
                glBufferSubData(GL_ARRAY_BUFFER, 0, self.particlesAfterExplosionTrailVertices.nbytes, self.particlesAfterExplosionTrailVertices)
                glBindBuffer(GL_ARRAY_BUFFER, 0)

                glBindVertexArray(self.VAO[3])
                glDrawArrays(GL_POINTS, 0, numCurrentParticlesAfterExplosion * self.numTrailParticles)

            glBindVertexArray(0)

        glPopAttrib()

    def _InitializeVertexArray(self):
        particlesBeforeExplosionVerticesData = []
        particlesBeforeExplosionTrailVerticesData = []
        particlesAfterExplosionVerticesData = []
        particlesAfterExplosionTrailVerticesData = []

        for i in range(1):
            particlesBeforeExplosionVerticesData.append(0.0)
            particlesBeforeExplosionVerticesData.append(0.0)
            particlesBeforeExplosionVerticesData.append(0.0)

            #particlesBeforeExplosionVerticesData.append(self.color[0])
            #particlesBeforeExplosionVerticesData.append(self.color[1])
            #particlesBeforeExplosionVerticesData.append(self.color[2])
            particlesBeforeExplosionVerticesData.append(random.random())
            particlesBeforeExplosionVerticesData.append(random.random())
            particlesBeforeExplosionVerticesData.append(random.random())
            particlesBeforeExplosionVerticesData.append(1.0)

            for j in range(self.numTrailParticles):
                particlesBeforeExplosionTrailVerticesData.append(0.0)
                particlesBeforeExplosionTrailVerticesData.append(0.0)
                particlesBeforeExplosionTrailVerticesData.append(0.0)

                #particlesBeforeExplosionTrailVerticesData.append(self.color[0])
                #particlesBeforeExplosionTrailVerticesData.append(self.color[1])
                #particlesBeforeExplosionTrailVerticesData.append(self.color[2])
                particlesBeforeExplosionTrailVerticesData.append(random.random())
                particlesBeforeExplosionTrailVerticesData.append(random.random())
                particlesBeforeExplosionTrailVerticesData.append(random.random())
                particlesBeforeExplosionTrailVerticesData.append(1.0)

        for i in range(self.numParticlesAfterExplosion):
            particlesAfterExplosionVerticesData.append(0.0)
            particlesAfterExplosionVerticesData.append(0.0)
            particlesAfterExplosionVerticesData.append(0.0)

            #particlesAfterExplosionVerticesData.append(self.color[0])
            #particlesAfterExplosionVerticesData.append(self.color[1])
            #particlesAfterExplosionVerticesData.append(self.color[2])
            particlesAfterExplosionVerticesData.append(random.random())
            particlesAfterExplosionVerticesData.append(random.random())
            particlesAfterExplosionVerticesData.append(random.random())
            particlesAfterExplosionVerticesData.append(1.0)

            for j in range(self.numTrailParticles):
                particlesAfterExplosionTrailVerticesData.append(0.0)
                particlesAfterExplosionTrailVerticesData.append(0.0)
                particlesAfterExplosionTrailVerticesData.append(0.0)

                particlesAfterExplosionTrailVerticesData.append(random.random())
                particlesAfterExplosionTrailVerticesData.append(random.random())
                particlesAfterExplosionTrailVerticesData.append(random.random())
                #particlesAfterExplosionTrailVerticesData.append(self.color[0])
                #particlesAfterExplosionTrailVerticesData.append(self.color[1])
                #particlesAfterExplosionTrailVerticesData.append(self.color[2])
                particlesAfterExplosionTrailVerticesData.append(1.0)

        self.particlesBeforeExplosionVertices = np.array(particlesBeforeExplosionVerticesData, dtype = np.float32)
        self.particlesBeforeExplosionTrailVertices = np.array(particlesBeforeExplosionTrailVerticesData, dtype = np.float32)
        self.particlesAfterExplosionVertices = np.array(particlesAfterExplosionVerticesData, dtype = np.float32)
        self.particlesAfterExplosionTrailVertices = np.array(particlesAfterExplosionTrailVerticesData, dtype = np.float32)

        glBindVertexArray(self.VAO[0])

        glBindBuffer(GL_ARRAY_BUFFER, self.VBO[0])
        glBufferData(GL_ARRAY_BUFFER, self.particlesBeforeExplosionVertices.nbytes, self.particlesBeforeExplosionVertices, GL_DYNAMIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.particlesBeforeExplosionVertices.itemsize * 7, ctypes.c_void_p(0))

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, self.particlesBeforeExplosionVertices.itemsize * 7, ctypes.c_void_p(self.particlesBeforeExplosionVertices.itemsize * 3))

        glBindVertexArray(self.VAO[1])

        glBindBuffer(GL_ARRAY_BUFFER, self.VBO[1])
        glBufferData(GL_ARRAY_BUFFER, self.particlesBeforeExplosionTrailVertices.nbytes, self.particlesBeforeExplosionTrailVertices, GL_DYNAMIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.particlesBeforeExplosionTrailVertices.itemsize * 7, ctypes.c_void_p(0))

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, self.particlesBeforeExplosionTrailVertices.itemsize * 7, ctypes.c_void_p(self.particlesBeforeExplosionTrailVertices.itemsize * 3))

        glBindVertexArray(self.VAO[2])

        glBindBuffer(GL_ARRAY_BUFFER, self.VBO[2])
        glBufferData(GL_ARRAY_BUFFER, self.particlesAfterExplosionVertices.nbytes, self.particlesAfterExplosionVertices, GL_DYNAMIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.particlesAfterExplosionVertices.itemsize * 7, ctypes.c_void_p(0))

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, self.particlesAfterExplosionVertices.itemsize * 7, ctypes.c_void_p(self.particlesAfterExplosionVertices.itemsize * 3))

        glBindVertexArray(self.VAO[3])

        glBindBuffer(GL_ARRAY_BUFFER, self.VBO[3])
        glBufferData(GL_ARRAY_BUFFER, self.particlesAfterExplosionTrailVertices.nbytes, self.particlesAfterExplosionTrailVertices, GL_DYNAMIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.particlesAfterExplosionTrailVertices.itemsize * 7, ctypes.c_void_p(0))

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, self.particlesAfterExplosionTrailVertices.itemsize * 7, ctypes.c_void_p(self.particlesAfterExplosionTrailVertices.itemsize * 3))

        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glBindVertexArray(0)

class Particle:
    def __init__(self, externalForece, pos, exploded, numTrailParticles):
        self.acceleration = externalForece

        self.lifespan = 1.0
        self.exploded = exploded

        self.trailPos = []
        self.numTrailParticles = numTrailParticles

        if self.exploded == False:
            self.velocity = glm.vec3(0.0, random.random() * 30.0 + 15.0, 0.0)
        else:
            ranVec = glm.vec3(random.random() * 2 - 1.0, random.random() * 2 - 1.0, random.random() * 2 - 1.0)
            ranVec = glm.normalize(ranVec)
            ranVec *= random.randrange(1, 50)
            self.velocity = glm.vec3(ranVec.x, ranVec.y, ranVec.z)

        self.pos = glm.vec3(pos.x, pos.y, pos.z)

        for i in range(self.numTrailParticles):
            self.trailPos.append(glm.vec3(pos.x, pos.y, pos.z))

    def GetVelocity(self):
        return self.velocity

    def GetPosition(self):
        return self.pos

    def GetLifespan(self):
        return self.lifespan

    def GetTrailPosition(self):
        return self.trailPos

    def Update(self, deltaTime):
        if self.exploded == True:
            self.velocity *= 0.85
            self.lifespan -= 0.02

        self.velocity += self.acceleration * deltaTime
        self.pos += self.velocity * deltaTime

        newTrailPos = glm.vec3(self.pos.x, self.pos.y, self.pos.z)

        self.trailPos.pop()
        self.trailPos.insert(0, newTrailPos)

    def Done(self):
        if self.lifespan < 0.0:
            return True
        else:
            return False

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
        if key == glfw.KEY_T:
            gInputManager.SetKeyState('T', True)

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

    glfwWindow = glfw.create_window(displaySize[0], displaySize[1], "Fireworks.Part 1", None, None)

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
    #gSceneManager.AddObject(Firework(glm.vec3(0.0, -20.0, 0.0)))

    prjMat = []
    viewMat = []

    lastElapsedTime = glfw.get_time()
    deltaTime = 0.0

    while glfw.window_should_close(glfwWindow) == False:
        glfw.poll_events()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        gSceneManager.AddObject(Firework(glm.vec3(0.0, -50.0, 0.0)))

        gSceneManager.Update(deltaTime)

        prjMat = gSceneManager.GetPerspectivePrjMat()
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