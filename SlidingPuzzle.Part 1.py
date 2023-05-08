
from OpenGL.GL import *
from OpenGL.GL.shaders import *
from PIL import Image 
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
layout(location = 2) in vec2 aTexCoord;

out vec4 color;
out vec2 texCoord;

uniform mat4 prjMat;
uniform mat4 viewMat;
uniform mat4 modelMat;

void main()
{
    color = aColor;
    texCoord = aTexCoord;

    gl_Position = prjMat * viewMat * vec4(aPos, 1.0);    
    // gl_Position = prjMat * viewMat * modelMat * vec4(aPos, 1.0);
    // gl_Position = vec4(aPos, 1.0);
}

"""

fragmentShaderCode = """

# version 330 core

in vec4 color;
in vec2 texCoord;

out vec4 fragColor;

uniform sampler2D sTexture;
uniform bool useTexture;

void main()
{
    if(useTexture)
    {
        fragColor = texture(sTexture, texCoord);
    }
    else
    {
        fragColor = color;
    }
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
        self.screenControlWithMouse = False

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

    def GetScreenControlWithMouse(self):
        return self.screenControlWithMouse

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
        self.font = Font('..\Resources\Fonts\comic.ttf', 14)
        #self.font = Font('..\Resources\Fonts\cour.ttf', 14)
        #self.font = Font('..\Resources\Fonts\HYNAMM.ttf', 14)
        self.font.MakeFontTextureWithGenList()

    def AddObject(self, object):
        self.objects.append(object)

    def UpdateAboutKeyInput(self):
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
            for i in range(numObjects):
                self.objects[i].Restart()
            gInputManager.SetKeyState('R', False)

        if gInputManager.GetKeyState(glfw.KEY_UP) == True:
            for i in range(numObjects):
                self.objects[i].UpdateAboutKeyInput(glfw.KEY_UP)
            gInputManager.SetKeyState(glfw.KEY_UP, False)
        elif gInputManager.GetKeyState(glfw.KEY_DOWN) == True:
            for i in range(numObjects):
                self.objects[i].UpdateAboutKeyInput(glfw.KEY_DOWN)
            gInputManager.SetKeyState(glfw.KEY_DOWN, False)

    def UpdateAboutMouseInput(self):
        numObjects = len(self.objects)       

        if gInputManager.GetMouseButtonClick(glfw.MOUSE_BUTTON_LEFT) == True:            
            lastMousePosOnClick = gInputManager.GetLastMousePosOnClick()
            for i in range(numObjects):
                self.objects[i].UpdateAboutMouseInput(glfw.MOUSE_BUTTON_LEFT, lastMousePosOnClick)
            gInputManager.SetMouseButtonClick(glfw.MOUSE_BUTTON_LEFT, False)

    def Update(self, deltaTime):
        self.UpdateAboutKeyInput()

        self.UpdateAboutMouseInput()

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

        self.mouseButtonClick = [False, False, False]
        self.lastMousePos = [-1, -1]
        self.lastMousePosOnClick = [-1, -1]

        self.keys = {}

    def GetMouseEntered(self):
        return self.mouseEntered

    def SetMouseEntered(self, value):
        self.mouseEntered = value

    def GetMouseButtonClick(self, button):
        return self.mouseButtonClick[button]

    def SetMouseButtonClick(self, button, value):
        self.mouseButtonClick[button] = value

    def GetLastMousePos(self):
        return self.lastMousePos

    def SetLastMousePos(self, value):
        self.lastMousePos = value    

    def GetLastMousePosOnClick(self):
        return self.lastMousePosOnClick

    def SetLastMousePosOnClick(self, value):
        self.lastMousePosOnClick = value

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

    def UpdateAboutKeyInput(self, key, value = True):
        pass

    def UpdateAboutMouseInput(self, button, pos):
        pass

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

    def SetBool(self, name, value):
        loc = glGetUniformLocation(self.program, name)
        
        glUniform1i(loc, value)

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

class SlidingPuzzle:
    def __init__(self, displaySize, shader):
        self.backgroundStuffVerticesList = []
        self.backgroundStuffIndicesList = []

        self.backgroundVertices = []
        self.backgroundIndices = []

        self.backgroundSubAreaVertices = []
        self.backgroundSubAreaIndices = []

        self.backgroundLineVertices =[]
        self.backgroundLineIndices = []

        self.backgroundSubAreaLineVertices = []
        self.backgroundSubAreaLineIndices = []

        self.backgroundBoardLineVertices = []
        self.backgroundBoardLineIndices = []

        self.gameStuffVerticesList = []
        self.gameStuffIndicesList = []

        self.allBlocksVertices = []
        self.allBlocksIndices = []

        self.boardSize = [4, 4]
        self.boardPos = []

        self.allBlocks = []
        self.allBlockColors = []
        self.allBlocksMovement = []

        self.blockSize = 140

        self.displaySize = (displaySize[0], displaySize[1])

        self.numShuffle = 500

        self.moveBlocks = []
        self.movingSpeed = 2000.0
        self.moving = False

        self.dirty = False

        self.texId = -1

        self.shader = shader
        self.useTextureInBackgroundStuff = []
        self.useTextureInGameStuff = []

        self.numBlockTypes = 7
        self.numVerticesOneBlock = 4
        self.numVertexComponents = 7
        self.numVertexComponentsWithTexture = 9
        self.numBackgroundStuff = 5
        self.numGameStuff = 2

        self.backgroundVAO = glGenVertexArrays(self.numBackgroundStuff)
        self.backgroundVBO = glGenBuffers(self.numBackgroundStuff)
        self.backgroundEBO = glGenBuffers(self.numBackgroundStuff)

        self.gameVAO = glGenVertexArrays(self.numGameStuff)
        self.gameVBO = glGenBuffers(self.numGameStuff)
        self.gameEBO = glGenBuffers(self.numGameStuff)

        self._InitializeBackgroundStuff()

        self._InitializeGameStuff()

        self._InitializeTexture()

    def Restart(self):
        self._InitializeBackgroundStuff()

        self._InitializeGameStuff()

    def UpdateAboutKeyInput(self, key, value = True):
        if key == glfw.KEY_UP:
            oldBoardSizeX = self.boardSize[0]

            self.boardSize[0] *= 2
            self.boardSize[1] *= 2
            self.blockSize /= 2

            if self.boardSize[0] > 16:
                self.boardSize[0] = 16
                self.boardSize[1] = 16
                self.blockSize = 35

            if oldBoardSizeX != self.boardSize[0]:
                self.Restart()

        elif key == glfw.KEY_DOWN:
            oldBoardSizeX = self.boardSize[0]

            self.boardSize[0] = int(self.boardSize[0] / 2)
            self.boardSize[1] = int(self.boardSize[1] / 2)
            self.blockSize *= 2

            if self.boardSize[0] < 4:
                self.boardSize[0] = 4
                self.boardSize[1] = 4
                self.blockSize = 140

            if oldBoardSizeX != self.boardSize[0]:
                self.Restart()

    def UpdateAboutMouseInput(self, button, pos):
        if button == glfw.MOUSE_BUTTON_LEFT and self.moving == False:
            if pos[0] < self.boardPos[0][0] or self.boardPos[1][0] < pos[0] or (self.displaySize[1] - pos[1]) < self.boardPos[0][1] or self.boardPos[1][1] < (self.displaySize[1] - pos[1]):
                return

            blockX = int((pos[0] - self.boardPos[0][0]) / self.blockSize)
            blockY = int((self.displaySize[1] - pos[1] - self.boardPos[0][1]) / self.blockSize)

            checkBlockX, checkBlockY = -1, -1
            findEmptyBlock = False

            self.moveBlocks.clear()

            if blockX != 0 and findEmptyBlock == False:
                checkBlockX, checkBlockY = blockX, blockY

                while 0 <= checkBlockX and self.allBlocks[checkBlockY][checkBlockX] != 255:
                    self.moveBlocks.insert(0, [checkBlockX, checkBlockY])

                    checkBlockX -= 1

                if 0 <= checkBlockX and self.allBlocks[checkBlockY][checkBlockX] == 255:
                    self.moveBlocks.insert(0, [checkBlockX, checkBlockY])

                    for i in range(len(self.moveBlocks) - 1):
                        self.allBlocksMovement[self.moveBlocks[i][1]][self.moveBlocks[i][0]][0] = self.blockSize

                    findEmptyBlock = True
                else:
                    self.moveBlocks.clear()

            if blockX != (self.boardSize[0] - 1) and findEmptyBlock == False:
                checkBlockX, checkBlockY = blockX, blockY

                while checkBlockX <= (self.boardSize[0] - 1) and self.allBlocks[checkBlockY][checkBlockX] != 255:
                    self.moveBlocks.insert(0, [checkBlockX, checkBlockY])

                    checkBlockX += 1

                if checkBlockX <= (self.boardSize[0] - 1) and self.allBlocks[checkBlockY][checkBlockX] == 255:
                    self.moveBlocks.insert(0, [checkBlockX, checkBlockY])

                    for i in range(len(self.moveBlocks) - 1):
                        self.allBlocksMovement[self.moveBlocks[i][1]][self.moveBlocks[i][0]][0] = -self.blockSize

                    findEmptyBlock = True
                else:
                    self.moveBlocks.clear()

            if blockY != 0 and findEmptyBlock == False:
                checkBlockX, checkBlockY = blockX, blockY

                while 0 <= checkBlockY and self.allBlocks[checkBlockY][checkBlockX] != 255:
                    self.moveBlocks.insert(0, [checkBlockX, checkBlockY])

                    checkBlockY -= 1

                if 0 <= checkBlockY and self.allBlocks[checkBlockY][checkBlockX] == 255:
                    self.moveBlocks.insert(0, [checkBlockX, checkBlockY])

                    for i in range(len(self.moveBlocks) - 1):
                        self.allBlocksMovement[self.moveBlocks[i][1]][self.moveBlocks[i][0]][1] = self.blockSize

                    findEmptyBlock = True
                else:
                    self.moveBlocks.clear()

            if blockY != (self.boardSize[1] - 1) and findEmptyBlock == False:
                checkBlockX, checkBlockY = blockX, blockY

                while checkBlockY <= (self.boardSize[1] - 1) and self.allBlocks[checkBlockY][checkBlockX] != 255:
                    self.moveBlocks.insert(0, [checkBlockX, checkBlockY])

                    checkBlockY += 1

                if checkBlockY <= (self.boardSize[1] - 1) and self.allBlocks[checkBlockY][checkBlockX] == 255:
                    self.moveBlocks.insert(0, [checkBlockX, checkBlockY])

                    for i in range(len(self.moveBlocks) - 1):
                        self.allBlocksMovement[self.moveBlocks[i][1]][self.moveBlocks[i][0]][1] = -self.blockSize

                    findEmptyBlock = True
                else:
                    self.moveBlocks.clear()

            if len(self.moveBlocks) > 0:
                for i in range(len(self.moveBlocks) - 1):
                    preBlockX, preBlockY = self.moveBlocks[i][0], self.moveBlocks[i][1]
                    curBlockX, curBlockY = self.moveBlocks[i + 1][0], self.moveBlocks[i + 1][1]

                    self.allBlocks[preBlockY][preBlockX] = self.allBlocks[curBlockY][curBlockX]

                lastBlockX, lastBlockY = self.moveBlocks[len(self.moveBlocks) - 1][0], self.moveBlocks[len(self.moveBlocks) - 1][1]

                self.allBlocks[lastBlockY][lastBlockX] = 255

                self.dirty = True

    def Update(self, deltaTime):
        self._UpdateMovingBlocks(deltaTime)

    def Draw(self):
        self._DrawBackground()

        self._DrawBoard()

    def _InitializeBackgroundStuff(self):
        boardLbPos = [20.0, 20.0]
        boardRtPos = []
        boardRtPos.append(boardLbPos[0] + self.boardSize[0] * self.blockSize)
        boardRtPos.append(boardLbPos[1] + self.boardSize[1] * self.blockSize)

        self.boardPos.append(boardLbPos)
        self.boardPos.append(boardRtPos)

        self.useTextureInBackgroundStuff = [False for i in range(self.numBackgroundStuff)]

        backgroundStuffIndex = 0

        backgroundVerticesData = [
            0.0, 0.0, -5.0, 1.0, 0.0, 0.0, 0.5,
            self.boardPos[0][0], 0.0, -5.0, 1.0, 0.0, 0.0, 0.5,
            self.boardPos[0][0], self.displaySize[1], -5.0, 1.0, 0.0, 0.0, 0.5,
            0.0, self.displaySize[1], -5.0, 1.0, 0.0, 0.0, 0.5,

            self.boardPos[1][0], 0.0, -5.0, 1.0, 0.0, 0.0, 0.5,
            self.displaySize[0], 0.0, -5.0, 1.0, 0.0, 0.0, 0.5,
            self.displaySize[0], self.displaySize[1], -5.0, 1.0, 0.0, 0.0, 0.5,
            self.boardPos[1][0], self.displaySize[1], -5.0, 1.0, 0.0, 0.0, 0.5,

            0.0, 0.0, -5.0, 1.0, 0.0, 0.0, 0.5,
            self.displaySize[0], 0.0, -5.0, 1.0, 0.0, 0.0, 0.5,
            self.displaySize[0], self.boardPos[0][1], -5.0, 1.0, 0.0, 0.0, 0.5,
            0.0, self.boardPos[0][1], -5.0, 1.0, 0.0, 0.0, 0.5,

            0.0, self.boardPos[1][1], -5.0, 1.0, 0.0, 0.0, 0.5,
            self.displaySize[0], self.boardPos[1][1], -5.0, 1.0, 0.0, 0.0, 0.5,
            self.displaySize[0], self.displaySize[1], -5.0, 1.0, 0.0, 0.0, 0.5,
            0.0, self.displaySize[1], -5.0, 1.0, 0.0, 0.0, 0.5
            ]

        backgroundIndicesData = [
            0, 1, 2, 2, 3, 0,
            4, 5, 6, 6, 7, 4,
            8, 9, 10, 10, 11, 8, 
            12, 13, 14, 14, 15, 12
            ]

        self.backgroundVertices = np.array(backgroundVerticesData, dtype = np.float32)
        self.backgroundIndices = np.array(backgroundIndicesData, dtype = np.uint32)

        backgroundStuffIndex += 1
        self.useTextureInBackgroundStuff[backgroundStuffIndex] = True

        backgroundSubAreaVerticesData = [
            600.0, 40.0, -4.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 
            780.0, 40.0, -4.5, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0,
            780.0, 220.0, -4.5, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
            600.0, 220.0, -4.5, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0
            ]

        backgroundSubAreaIndicesData = [
            0, 1, 2, 2, 3, 0
            ]

        self.backgroundSubAreaVertices = np.array(backgroundSubAreaVerticesData, dtype = np.float32)
        self.backgroundSubAreaIndices = np.array(backgroundSubAreaIndicesData, dtype = np.uint32)

        backgroundStuffIndex += 1

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

        backgroundStuffIndex += 1

        backgroundSubAreaLineVerticesData = [
            backgroundSubAreaVerticesData[0], backgroundSubAreaVerticesData[1], -4.0, 1.0, 1.0, 1.0, 1.0,
            backgroundSubAreaVerticesData[9], backgroundSubAreaVerticesData[10], -4.0, 1.0, 1.0, 1.0, 1.0,
            backgroundSubAreaVerticesData[18], backgroundSubAreaVerticesData[19], -4.0, 1.0, 1.0, 1.0, 1.0,
            backgroundSubAreaVerticesData[27], backgroundSubAreaVerticesData[28], -4.0, 1.0, 1.0, 1.0, 1.0
            ]

        backgroundSubAreaLineIndicesData = [
            0, 1, 1, 2, 2, 3, 3, 0
            ]

        self.backgroundSubAreaLineVertices = np.array(backgroundSubAreaLineVerticesData, dtype = np.float32)
        self.backgroundSubAreaLineIndices = np.array(backgroundSubAreaLineIndicesData, dtype = np.uint32)

        backgroundStuffIndex += 1

        backgroundBoardLineVerticesData = []
        backgroundBoardLineIndicesData = []

        boardLineIndex = 0

        for i in range(self.boardSize[0]):
            lineX = self.boardPos[0][0] + (self.blockSize * (i + 1))

            backgroundBoardLineVerticesData.append(lineX)
            backgroundBoardLineVerticesData.append(self.boardPos[0][1])
            backgroundBoardLineVerticesData.append(1.0)

            backgroundBoardLineVerticesData.append(1.0)
            backgroundBoardLineVerticesData.append(1.0)
            backgroundBoardLineVerticesData.append(1.0)
            backgroundBoardLineVerticesData.append(1.0)

            backgroundBoardLineVerticesData.append(lineX)
            backgroundBoardLineVerticesData.append(self.boardPos[1][1])
            backgroundBoardLineVerticesData.append(1.0)

            backgroundBoardLineVerticesData.append(1.0)
            backgroundBoardLineVerticesData.append(1.0)
            backgroundBoardLineVerticesData.append(1.0)
            backgroundBoardLineVerticesData.append(1.0)

            backgroundBoardLineIndicesData.append(2 * boardLineIndex + 0)
            backgroundBoardLineIndicesData.append(2 * boardLineIndex + 1)

            boardLineIndex += 1

        for i in range(self.boardSize[1]):
            lineY = self.boardPos[0][1] + (self.blockSize * (i + 1))

            backgroundBoardLineVerticesData.append(self.boardPos[0][0])
            backgroundBoardLineVerticesData.append(lineY)
            backgroundBoardLineVerticesData.append(1.0)

            backgroundBoardLineVerticesData.append(1.0)
            backgroundBoardLineVerticesData.append(1.0)
            backgroundBoardLineVerticesData.append(1.0)
            backgroundBoardLineVerticesData.append(1.0)

            backgroundBoardLineVerticesData.append(self.boardPos[1][0])
            backgroundBoardLineVerticesData.append(lineY)
            backgroundBoardLineVerticesData.append(1.0)

            backgroundBoardLineVerticesData.append(1.0)
            backgroundBoardLineVerticesData.append(1.0)
            backgroundBoardLineVerticesData.append(1.0)
            backgroundBoardLineVerticesData.append(1.0)

            backgroundBoardLineIndicesData.append(2 * boardLineIndex + 0)
            backgroundBoardLineIndicesData.append(2 * boardLineIndex + 1)

            boardLineIndex += 1

        self.backgroundBoardLineVertices = np.array(backgroundBoardLineVerticesData, dtype = np.float32)
        self.backgroundBoardLineIndices = np.array(backgroundBoardLineIndicesData, dtype = np.uint32)

        self.backgroundStuffVerticesList.clear()
        self.backgroundStuffIndicesList.clear()

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

            numVertexComponents = 0

            if self.useTextureInBackgroundStuff[i] == True:
                numVertexComponents = self.numVertexComponentsWithTexture

                glEnableVertexAttribArray(2)
                glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, self.backgroundStuffVerticesList[i].itemsize * numVertexComponents, ctypes.c_void_p(self.backgroundStuffVerticesList[i].itemsize * 7))
            else:
                numVertexComponents = self.numVertexComponents

            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.backgroundStuffVerticesList[i].itemsize * numVertexComponents, ctypes.c_void_p(0))

            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, self.backgroundStuffVerticesList[i].itemsize * numVertexComponents, ctypes.c_void_p(self.backgroundStuffVerticesList[i].itemsize * 3))

        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glBindVertexArray(0)

    def _ShuffleAllBlocks(self, allBlocksVerticesData):
        numDirections = 4
        curBlockX, curBlockY = self.boardSize[0] - 1, 0
        nextBlockX, nextBlockY = curBlockX, curBlockY

        for i in range(self.numShuffle):
            decision = False

            while decision == False:
                direction = random.randrange(0, numDirections)

                if direction == 0:
                    nextBlockY -= 1
                elif direction == 1:
                    nextBlockX += 1
                elif direction == 2:
                    nextBlockY += 1
                elif direction == 3:
                    nextBlockX -= 1

                decision = True

                if nextBlockX < 0 or (self.boardSize[0] - 1) < nextBlockX:
                    decision = False
                elif nextBlockY < 0 or (self.boardSize[1] - 1) < nextBlockY:
                    decision = False

                if decision == False:
                    nextBlockX, nextBlockY = curBlockX, curBlockY

            curBROffset = curBlockY * self.boardSize[0] * self.numVerticesOneBlock * self.numVertexComponentsWithTexture
            curBCOffset = curBlockX * self.numVerticesOneBlock * self.numVertexComponentsWithTexture

            nextBROffset = nextBlockY * self.boardSize[0] * self.numVerticesOneBlock * self.numVertexComponentsWithTexture
            nextBCOffset = nextBlockX * self.numVerticesOneBlock * self.numVertexComponentsWithTexture

            for v in range(self.numVerticesOneBlock):
                vOffset = v * self.numVertexComponentsWithTexture

                allBlocksVerticesData[curBROffset + curBCOffset + vOffset + 7] = allBlocksVerticesData[nextBROffset + nextBCOffset + vOffset + 7]
                allBlocksVerticesData[curBROffset + curBCOffset + vOffset + 8] = allBlocksVerticesData[nextBROffset + nextBCOffset + vOffset + 8]

            curBlockX, curBlockY = nextBlockX, nextBlockY

        tmp = self.allBlocks[curBlockY][curBlockX]
        self.allBlocks[curBlockY][curBlockX] = 255
        self.allBlocks[0][self.boardSize[0] - 1] = tmp

    def _InitializeGameStuff(self):
        self.allBlocks = np.full((self.boardSize[1], self.boardSize[0]), fill_value = 255)
        self.allBlocksMovement = np.full((self.boardSize[1], self.boardSize[0], 2), fill_value = 0.0)

        self.useTextureInGameStuff = [False for i in range(self.numGameStuff)]

        gameStuffIndex = 0

        self.useTextureInGameStuff[gameStuffIndex] = True

        self.allBlockColors.clear()

        self.allBlockColors.append([1.0, 0.0, 0.0])
        self.allBlockColors.append([0.0, 0.76, 0.0])
        self.allBlockColors.append([0.15, 0.18, 0.85])
        self.allBlockColors.append([0.9, 0.73, 0.0])
        self.allBlockColors.append([0.95, 0.0, 0.89])
        self.allBlockColors.append([0.0, 0.9, 0.91])
        self.allBlockColors.append([1.0, 0.56, 0.0])

        allBlocksVerticesData = []
        allBlocksIndicesData = []

        texCoordInterval = 1.0  / self.boardSize[0]

        for r in range(self.boardSize[1]):
            for c in range(self.boardSize[0]):
                self.allBlocks[r][c] = random.randrange(0, self.numBlockTypes)

                for v in range(self.numVerticesOneBlock):
                    allBlocksVerticesData.append(0.0)
                    allBlocksVerticesData.append(0.0)
                    allBlocksVerticesData.append(0.0)

                    allBlocksVerticesData.append(1.0)
                    allBlocksVerticesData.append(1.0)
                    allBlocksVerticesData.append(1.0)
                    allBlocksVerticesData.append(1.0)

                    if self.useTextureInGameStuff[gameStuffIndex] == True:
                        if v == 0:
                            allBlocksVerticesData.append(c * texCoordInterval)
                            allBlocksVerticesData.append(r * texCoordInterval)
                        elif v == 1:
                            allBlocksVerticesData.append((c + 1) * texCoordInterval)
                            allBlocksVerticesData.append(r * texCoordInterval)
                        elif v == 2:
                            allBlocksVerticesData.append((c + 1) * texCoordInterval)
                            allBlocksVerticesData.append((r + 1) * texCoordInterval)
                        elif v == 3:
                            allBlocksVerticesData.append(c * texCoordInterval)
                            allBlocksVerticesData.append((r + 1) * texCoordInterval)

        self.allBlocks[0][self.boardSize[0] - 1] = 255

        for r in range(self.boardSize[1]):
            rOffset = r * self.boardSize[0] * self.numVerticesOneBlock

            for c in range(self.boardSize[0]):
                cOffset = c * self.numVerticesOneBlock

                allBlocksIndicesData.append(rOffset + cOffset + 0)
                allBlocksIndicesData.append(rOffset + cOffset + 1)
                allBlocksIndicesData.append(rOffset + cOffset + 2)

                allBlocksIndicesData.append(rOffset + cOffset + 2)
                allBlocksIndicesData.append(rOffset + cOffset + 3)
                allBlocksIndicesData.append(rOffset + cOffset + 0)

        if self.useTextureInGameStuff[gameStuffIndex] == True:
            self._ShuffleAllBlocks(allBlocksVerticesData)

        self.allBlocksVertices = np.array(allBlocksVerticesData, dtype = np.float32)
        self.allBlocksIndices = np.array(allBlocksIndicesData, dtype = np.uint32)

        self.gameStuffVerticesList.clear()
        self.gameStuffIndicesList.clear()

        self.gameStuffVerticesList.append(self.allBlocksVertices)

        self.gameStuffIndicesList.append(self.allBlocksIndices)

        for i in range(self.numGameStuff - 1):
            glBindVertexArray(self.gameVAO[i])

            glBindBuffer(GL_ARRAY_BUFFER, self.gameVBO[i])
            glBufferData(GL_ARRAY_BUFFER, self.gameStuffVerticesList[i].nbytes, self.gameStuffVerticesList[i], GL_DYNAMIC_DRAW)

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.gameEBO[i])
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.gameStuffIndicesList[i].nbytes, self.gameStuffIndicesList[i], GL_STATIC_DRAW)

            numVertexComponents = 0

            if self.useTextureInGameStuff[i] == True:
                numVertexComponents = self.numVertexComponentsWithTexture

                glEnableVertexAttribArray(2)
                glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, self.gameStuffVerticesList[i].itemsize * numVertexComponents, ctypes.c_void_p(self.gameStuffVerticesList[i].itemsize * 7))
            else:
                numVertexComponents = self.numVertexComponents

            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.gameStuffVerticesList[i].itemsize * numVertexComponents, ctypes.c_void_p(0))

            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, self.gameStuffVerticesList[i].itemsize * numVertexComponents, ctypes.c_void_p(self.gameStuffVerticesList[i].itemsize * 3))

        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glBindVertexArray(0)

    def _InitializeTexture(self):
        #image = Image.open('..\Resources\Textures\Praha.1024.jpg')
        image = Image.open('..\Resources\Textures\WorldMap.1024.jpg')
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        self.texId = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texId)

        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, image.tobytes())

        glBindTexture(GL_TEXTURE_2D, 0)

    def _UpdateMovingBlocks(self, deltaTime):
        self.moving = False

        for r in range(self.boardSize[1]):
            for c in range(self.boardSize[0]):
                if self.allBlocksMovement[r][c][0] != 0:
                    beforeMovement = self.allBlocksMovement[r][c][0]
                    self.allBlocksMovement[r][c][0] -= (self.allBlocksMovement[r][c][0] / abs(self.allBlocksMovement[r][c][0]) * deltaTime * self.movingSpeed)
                    afterMovement = self.allBlocksMovement[r][c][0]

                    if beforeMovement * afterMovement < 0.0:
                        self.allBlocksMovement[r][c][0] = 0.0

                    self.moving = True

                elif self.allBlocksMovement[r][c][1] != 0:
                    beforeMovement = self.allBlocksMovement[r][c][1]
                    self.allBlocksMovement[r][c][1] -= (self.allBlocksMovement[r][c][1] / abs(self.allBlocksMovement[r][c][1]) * deltaTime * self.movingSpeed)
                    afterMovement = self.allBlocksMovement[r][c][1]

                    if beforeMovement * afterMovement < 0.0:
                        self.allBlocksMovement[r][c][1] = 0.0

                    self.moving = True

    def _DrawBackground(self):
        glPushAttrib(GL_COLOR_BUFFER_BIT | GL_ENABLE_BIT | GL_LINE_BIT)

        glEnable(GL_BLEND)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.shader.SetBool('useTexture', False)

        backgroundStuffIndex = 0

        glBindVertexArray(self.backgroundVAO[backgroundStuffIndex])
        glDrawElements(GL_TRIANGLES, len(self.backgroundStuffIndicesList[backgroundStuffIndex]), GL_UNSIGNED_INT, None)

        backgroundStuffIndex += 1

        self.shader.SetBool('useTexture', self.useTextureInBackgroundStuff[backgroundStuffIndex])

        if self.useTextureInBackgroundStuff[backgroundStuffIndex] == True:
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, self.texId)

        glBindVertexArray(self.backgroundVAO[backgroundStuffIndex])
        glDrawElements(GL_TRIANGLES, len(self.backgroundStuffIndicesList[backgroundStuffIndex]), GL_UNSIGNED_INT, None)

        if self.useTextureInBackgroundStuff[backgroundStuffIndex] == True:
            glDisable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, 0)

        backgroundStuffIndex += 1

        self.shader.SetBool('useTexture', self.useTextureInBackgroundStuff[backgroundStuffIndex])

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

    def _DrawBoard(self):
        glPushAttrib(GL_COLOR_BUFFER_BIT | GL_ENABLE_BIT)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        gameStuffIndex = 0
        numVertexComponents = 0

        self.shader.SetBool('useTexture', self.useTextureInGameStuff[gameStuffIndex])

        if self.useTextureInGameStuff[gameStuffIndex] == True:
            numVertexComponents = self.numVertexComponentsWithTexture

            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, self.texId)
        else:
            numVertexComponents = self.numVertexComponents

        if self.dirty == True:
            for i in range(len(self.moveBlocks) - 1):
                preBlockX, preBlockY = self.moveBlocks[i][0], self.moveBlocks[i][1]
                curBlockX, curBlockY = self.moveBlocks[i + 1][0], self.moveBlocks[i + 1][1]

                preBlockROffset = preBlockY * self.boardSize[0] * self.numVerticesOneBlock * numVertexComponents
                preBlockCOffset = preBlockX * self.numVerticesOneBlock * numVertexComponents

                curBlockROffset = curBlockY * self.boardSize[0] * self.numVerticesOneBlock * numVertexComponents
                curBlockCOffset = curBlockX * self.numVerticesOneBlock * numVertexComponents

                for v in range(self.numVerticesOneBlock):
                    vOffset = v * numVertexComponents

                    if self.useTextureInGameStuff[gameStuffIndex] == True:
                        self.allBlocksVertices[preBlockROffset + preBlockCOffset + vOffset + 7] = self.allBlocksVertices[curBlockROffset + curBlockCOffset + vOffset + 7]
                        self.allBlocksVertices[preBlockROffset + preBlockCOffset + vOffset + 8] = self.allBlocksVertices[curBlockROffset + curBlockCOffset + vOffset + 8]

            lastBlockX, lastBlockY = self.moveBlocks[len(self.moveBlocks) - 1][0], self.moveBlocks[len(self.moveBlocks) - 1][1]

            lastBlockROffset = lastBlockY * self.boardSize[0] * self.numVerticesOneBlock * numVertexComponents
            lastBlockCOffset = lastBlockX * self.numVerticesOneBlock * numVertexComponents

            for v in range(self.numVerticesOneBlock):
                vOffset = v * numVertexComponents

                if self.useTextureInGameStuff[gameStuffIndex] == True:
                    self.allBlocksVertices[lastBlockROffset + lastBlockCOffset + vOffset + 0] = 0.0
                    self.allBlocksVertices[lastBlockROffset + lastBlockCOffset + vOffset + 1] = 0.0
                    self.allBlocksVertices[lastBlockROffset + lastBlockCOffset + vOffset + 2] = 0.0

            self.dirty = False

        for r in range(self.boardSize[1]):
            rOffset = r * self.boardSize[0] * self.numVerticesOneBlock * numVertexComponents

            for c in range(self.boardSize[0]):
                if self.allBlocks[r][c] != 255:
                    cOffset = c * self.numVerticesOneBlock * numVertexComponents

                    posX = self.boardPos[0][0] + (c + 1) * self.blockSize
                    posY = self.boardPos[0][1] + (r + 1) * self.blockSize

                    movePosX = posX + self.allBlocksMovement[r][c][0]
                    movePosY = posY + self.allBlocksMovement[r][c][1]

                    verticesIndex = 0
                    vOffset = verticesIndex * numVertexComponents

                    self.allBlocksVertices[rOffset + cOffset + vOffset + 0] = movePosX - self.blockSize
                    self.allBlocksVertices[rOffset + cOffset + vOffset + 1] = movePosY - self.blockSize
                    self.allBlocksVertices[rOffset + cOffset + vOffset + 2] = 0.0

                    verticesIndex += 1
                    vOffset = verticesIndex * numVertexComponents

                    self.allBlocksVertices[rOffset + cOffset + vOffset + 0] = movePosX
                    self.allBlocksVertices[rOffset + cOffset + vOffset + 1] = movePosY - self.blockSize
                    self.allBlocksVertices[rOffset + cOffset + vOffset + 2] = 0.0

                    verticesIndex += 1
                    vOffset = verticesIndex * numVertexComponents

                    self.allBlocksVertices[rOffset + cOffset + vOffset + 0] = movePosX
                    self.allBlocksVertices[rOffset + cOffset + vOffset + 1] = movePosY
                    self.allBlocksVertices[rOffset + cOffset + vOffset + 2] = 0.0

                    verticesIndex += 1
                    vOffset = verticesIndex * numVertexComponents

                    self.allBlocksVertices[rOffset + cOffset + vOffset + 0] = movePosX - self.blockSize
                    self.allBlocksVertices[rOffset + cOffset + vOffset + 1] = movePosY
                    self.allBlocksVertices[rOffset + cOffset + vOffset + 2] = 0.0

        glBindBuffer(GL_ARRAY_BUFFER, self.gameVBO[gameStuffIndex])
        glBufferSubData(GL_ARRAY_BUFFER, 0, self.gameStuffVerticesList[gameStuffIndex].nbytes, self.gameStuffVerticesList[gameStuffIndex])
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glBindVertexArray(self.gameVAO[gameStuffIndex])
        glDrawElements(GL_TRIANGLES, len(self.gameStuffIndicesList[gameStuffIndex]), GL_UNSIGNED_INT, None)

        glBindTexture(GL_TEXTURE_2D, 0)

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

    if action == glfw.RELEASE:
        if key == glfw.KEY_W:
            gInputManager.SetKeyState('W', False)
        elif key == glfw.KEY_S:
            gInputManager.SetKeyState('S', False)
        elif key == glfw.KEY_A:
            gInputManager.SetKeyState('A', False)
        elif key == glfw.KEY_D:
            gInputManager.SetKeyState('D', False)

def HandleMouseButtonCallback(glfwWindow, button, action, mod):
    if button == glfw.MOUSE_BUTTON_LEFT:
        if action == glfw.PRESS:            
            gInputManager.SetMouseButtonClick(glfw.MOUSE_BUTTON_LEFT, True)
            gInputManager.SetLastMousePosOnClick(glfw.get_cursor_pos(glfwWindow))

def HandleCursorPosCallback(glfwWindow, xPos, yPos):
    if gSceneManager.GetPause() == True:
        return

    if gSceneManager.GetScreenControlWithMouse() == False:
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

    glfwWindow = glfw.create_window(displaySize[0], displaySize[1], "SlidingPuzzle.Part 1", None, None)

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

    glfw.make_context_current(glfwWindow)

    glfw.set_window_size_callback(glfwWindow, HandleWindowSizeCallback)

    glfw.set_key_callback(glfwWindow, HandleKeyCallback) 

    glfw.set_mouse_button_callback(glfwWindow, HandleMouseButtonCallback)
    
    glfw.set_cursor_pos_callback(glfwWindow, HandleCursorPosCallback)

    shader = Shader(vertexShaderCode, fragmentShaderCode)

    gSceneManager.InitializeOpenGL()
    gSceneManager.SetCamera(Camera())
    gSceneManager.MakeFont()
    gSceneManager.AddObject(SlidingPuzzle(displaySize, shader))

    prjMat = []
    viewMat = []

    lastElapsedTime = glfw.get_time()
    deltaTime = 0.0

    while glfw.window_should_close(glfwWindow) == False:
        glfw.poll_events()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        gSceneManager.Update(deltaTime)

        if gSceneManager.GetView3D() == True:
            prjMat = gSceneManager.GetPerspectivePrjMat()
        else:
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