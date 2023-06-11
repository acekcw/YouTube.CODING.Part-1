
from turtle import circle
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
    gl_Position = prjMat * viewMat * modelMat * vec4(aPos, 1.0);

    color = aColor;    
    
    // gl_Position = prjMat * viewMat * vec4(aPos, 1.0);
    // gl_Position = vec4(aPos, 1.0);
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


class SceneManager:
    def __init__(self, view3D):
        self.displaySize = [800, 600]

        self.programInfoAreaVertices = []
        self.programInfoAreaIndices = []

        self.fovy = 45.0
        self.aspect = self.displaySize[0] / self.displaySize[1]
        self.near = 0.1
        self.far = 1000.0

        self.shader = None

        self.camera = None
        self.sailingCamera = [False, False]

        self.perspectivePrjMat = glm.mat4()
        self.orthoPrjMat = glm.mat4()
        self.viewMat = glm.mat4()
        self.modelMat = glm.mat4()
        self.view3D = view3D

        self.screenControlWithMouse = False
        self.ableMouseDragged = False
        self.ableMouseEscape = True

        self.objects = []
        self.font = None

        self.deltaTime = 0.0
        self.dirty = True

        self.colors = {}

        self.programInfo = False
        self.numProgramInfoElement = 8

        self.pause = False
        self.debug = False
        self.debugMat = glm.mat4()

        self._InitializeColors()
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

    def GetAbleMouseDragged(self):
        return self.ableMouseDragged

    def GetAbleMouseEscape(self):
        return self.ableMouseEscape

    def GetColor(self, key, index):
        completedKey = key + str(index)
        return self.colors[completedKey]

    def GetShader(self):
        return self.shader

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

    def InitializeOpenGL(self, shader):
        self.shader = shader        

        color = self.GetColor('DefaultColor_', 1)
        glClearColor(color[0], color[1], color[2], 1.0)

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

        if gInputManager.GetKeyState(glfw.KEY_SPACE) == True:
            for i in range(numObjects):
                self.objects[i].UpdateAboutKeyInput(glfw.KEY_SPACE)
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
        if gInputManager.GetKeyState('E') == True:
            self.ableMouseEscape = not self.ableMouseEscape
            gInputManager.SetKeyState('E', False)
        if gInputManager.GetKeyState('G') == True:
            self.ableMouseDragged = not self.ableMouseDragged
            gInputManager.SetKeyState('G', False)
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
        if gInputManager.GetKeyState('S') == True:
            self.screenControlWithMouse = not self.screenControlWithMouse            
            gInputManager.SetKeyState('S', False)

        if gInputManager.GetKeyState(glfw.KEY_DOWN) == True:
            for i in range(numObjects):
                self.objects[i].UpdateAboutKeyInput(glfw.KEY_DOWN)
            gInputManager.SetKeyState(glfw.KEY_DOWN, False)

        if gInputManager.GetKeyState(glfw.KEY_LEFT) == True:
            for i in range(numObjects):
                self.objects[i].UpdateAboutKeyInput(glfw.KEY_LEFT)            
        if gInputManager.GetKeyState(glfw.KEY_RIGHT) == True:
            for i in range(numObjects):
                self.objects[i].UpdateAboutKeyInput(glfw.KEY_RIGHT)  
        if gInputManager.GetKeyState(glfw.KEY_UP) == True:            
            for i in range(numObjects):
                self.objects[i].UpdateAboutKeyInput(glfw.KEY_UP) 

    def UpdateAboutMouseInput(self):
        numObjects = len(self.objects)       

        if gInputManager.GetMouseButtonClick(glfw.MOUSE_BUTTON_LEFT) == True:            
            lastMousePosOnClick = gInputManager.GetLastMousePosOnClick()
            for i in range(numObjects):
                self.objects[i].UpdateAboutMouseInput(glfw.MOUSE_BUTTON_LEFT, lastMousePosOnClick)
        if gSceneManager.ableMouseDragged == False:
            gInputManager.SetMouseButtonClick(glfw.MOUSE_BUTTON_LEFT, False)

    def Update(self, deltaTime):
        self.shader.Use()

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

        if self.view3D == True:
            prjMat = self.perspectivePrjMat
        else:
            prjMat = self.orthoPrjMat

        self.shader.SetMat4('prjMat', prjMat)
        self.shader.SetMat4('viewMat', self.viewMat)
        self.shader.SetMat4('modelMat', self.modelMat)

        for i in range(numObjects):
            self.objects[i].Draw()

    def DrawProgramInfo(self, deltaTime):
        if self.programInfo == False:
            return

        glUseProgram(0)

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

    def _InitializeColors(self):
        self.colors['DefaultColor_0'] = [1.0, 1.0, 1.0]
        self.colors['DefaultColor_1'] = [0.0, 0.0, 0.0]
        self.colors['DefaultColor_2'] = [1.0, 0.0, 0.0]
        self.colors['DefaultColor_3'] = [0.0, 1.0, 0.0]
        self.colors['DefaultColor_4'] = [0.0, 0.0, 1.0]
        self.colors['DefaultColor_5'] = [0.8, 0.3, 0.5]
        self.colors['DefaultColor_6'] = [0.3, 0.8, 0.5]
        self.colors['DefaultColor_7'] = [0.2, 0.3, 0.98]        

        self.colors['ObjectColor_0'] = [1.0, 0.0, 0.0]
        self.colors['ObjectColor_1'] = [0.0, 0.76, 0.0]
        self.colors['ObjectColor_2'] = [0.15, 0.18, 0.85]
        self.colors['ObjectColor_3'] = [0.9, 0.73, 0.0]
        self.colors['ObjectColor_4'] = [0.95, 0.0, 0.89]
        self.colors['ObjectColor_5'] = [0.0, 0.9, 0.91]
        self.colors['ObjectColor_6'] = [1.0, 0.56, 0.0]

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
        
        color = self.GetColor('DefaultColor_', 6)
        glColor(color[0], color[1], color[2], 1.0)

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

        infoText.append('.SCWithMouse(S): ')
        infoTextIndex += 1
        
        if self.screenControlWithMouse == True:
            infoText[infoTextIndex] += "On"
        else:
            infoText[infoTextIndex] += "Off"

        infoText.append('.  Dragged(G): ')
        infoTextIndex += 1
        
        if self.ableMouseDragged == True:
            infoText[infoTextIndex] += "On"
        else:
            infoText[infoTextIndex] += "Off"

        infoText.append('.MouseEscape(E): ')
        infoTextIndex += 1
        
        if self.ableMouseEscape == True:
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

            if i == self.numProgramInfoElement - 2:                
                textPosY -= 80.0
            else:
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

    def GetMouseButtonClick(self, key):
        return self.mouseButtonClick[key]

    def SetMouseButtonClick(self, key, value):
        self.mouseButtonClick[key] = value

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

        self.numVertexComponents = 9

        self.VAO = glGenVertexArrays(1)
        self.VBO = glGenBuffers(1)
        self.EBO = glGenBuffers(1)

        glBindVertexArray(self.VAO)

        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.vertices.itemsize * self.numVertexComponents, ctypes.c_void_p(0))

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, self.vertices.itemsize * self.numVertexComponents, ctypes.c_void_p(self.vertices.itemsize * 3))

        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, self.vertices.itemsize * self.numVertexComponents, ctypes.c_void_p(self.vertices.itemsize * 7))

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0)

    def Restart(self):
        return

    def UpdateAboutKeyInput(self, key, value = True):
        pass

    def UpdateAboutMouseInput(self, button, pos):
        pass

    def Update(self, deltaTime):
        self.rotDegree += deltaTime * 50

        if self.rotDegree > 360.0:
            self.rotDegree = 0.0

    def Draw(self):
        glPushAttrib(GL_COLOR_BUFFER_BIT | GL_ENABLE_BIT)

        glEnable(GL_BLEND)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        transMat = glm.translate(glm.vec3(0.0, 0.0, 0.0))
        
        rotXMat = glm.rotate(glm.radians(self.rotDegree), glm.vec3(1.0, 0.0, 0.0))
        rotYMat = glm.rotate(glm.radians(self.rotDegree), glm.vec3(0.0, 1.0, 0.0))
        rotZMat = glm.rotate(glm.radians(self.rotDegree), glm.vec3(0.0, 0.0, 1.0))       

        scaleMat = glm.scale(glm.vec3(1.0, 1.0, 1.0))

        self.modelMat = transMat * rotZMat * rotYMat * rotXMat * scaleMat        

        shader = gSceneManager.GetShader()
        shader.SetMat4('modelMat', self.modelMat)

        glBindVertexArray(self.VAO)
        glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)

        glBindVertexArray(0)

        glPopAttrib()

    def _GenerateCube(self):
        cubeVerticesData = [            
            # Front
            -0.5, -0.5, 0.5, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.5, -0.5, 0.5, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0,
            0.5, 0.5, 0.5, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0,
            -0.5, 0.5, 0.5, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0,

            # Back
            0.5, -0.5, -0.5, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
            -0.5, -0.5, -0.5, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0,
            -0.5, 0.5, -0.5, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0,
            0.5, 0.5, -0.5, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,

            # Left
            -0.5, -0.5, -0.5, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
            -0.5, -0.5, 0.5, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0,
            -0.5, 0.5, 0.5, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
            -0.5, 0.5, -0.5, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0,

            # Right
            0.5, -0.5, 0.5, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0,
            0.5, -0.5, -0.5, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0,
            0.5, 0.5, -0.5, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0,
            0.5, 0.5, 0.5, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0,

            # Top
            -0.5, 0.5, 0.5, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0,
            0.5, 0.5, 0.5, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0,
            0.5, 0.5, -0.5, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            -0.5, 0.5, -0.5, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0,

            # Bottom
            -0.5, -0.5, -0.5, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0,
            0.5, -0.5, -0.5, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0,
            0.5, -0.5, 0.5, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0,
            -0.5, -0.5, 0.5, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0
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

class Ship:
    def __init__(self, boardPos):
        self.vertices = []
        self.indices = []

        self.acc = glm.vec3()
        self.vel = glm.vec3()
        self.pos = glm.vec3()

        self.headingAngle = 0.0
        self.headingVec = glm.vec3(0.0, 1.0, 0.0)

        self.modelMat = glm.mat4()        

        self.radius = 20

        self.immortal = True
        self.immortalElapsedTime = 0.0
        self.immortalTimeLimit = 3.0

        self.boardPos = boardPos

        self.dirty = False

        self.numVertexComponents = 7
        self.numDrawingStuff = 1

        self.drawingStuffVAO = glGenVertexArrays(self.numDrawingStuff)
        self.drawingStuffVBO = glGenBuffers(self.numDrawingStuff)
        self.drawingStuffEBO = glGenBuffers(self.numDrawingStuff)

        self._Initialize()

    def GetPos(self):
        return self.pos

    def GetHeadingVec(self):
        return self.headingVec

    def GetRadius(self):
        return self.radius

    def SetImmortal(self, value):
        self.immortal = value

    def CheckHit(self, asteroid):
        if self.immortal == True:
            return False

        dis = glm.length(self.pos - asteroid.GetPos())

        if dis < self.radius + asteroid.GetRadius():
            return True

        return False

    def UpdateAboutKeyInput(self, key, value = True):
        if key == glfw.KEY_LEFT:
            self.headingAngle += 5.0
            rotMat = glm.rotate(glm.radians(self.headingAngle), glm.vec3(0.0, 0.0, 1.0))
            self.headingVec = rotMat * glm.vec3(0.0, 1.0, 0.0)

        elif key == glfw.KEY_RIGHT:
            self.headingAngle -= 5.0
            rotMat = glm.rotate(glm.radians(self.headingAngle), glm.vec3(0.0, 0.0, 1.0))
            self.headingVec = rotMat * glm.vec3(0.0, 1.0, 0.0)

        elif key == glfw.KEY_UP:            
            self.acc = self.headingVec * 1000
            self.dirty = True

    def Update(self, deltaTime):
        if self.dirty == True:
            self.vel += self.acc * deltaTime

            self.dirty = False

        self.pos += self.vel * deltaTime
        self.vel *= 0.99

        self._CheckOffScreen()

        if self.immortal == True:
            self.immortalElapsedTime += deltaTime

            if self.immortalElapsedTime > self.immortalTimeLimit:
                self.immortal = False
                self.immortalElapsedTime = 0.0

    def Draw(self):
        glPushAttrib(GL_COLOR_BUFFER_BIT | GL_ENABLE_BIT | GL_POLYGON_BIT | GL_LINE_BIT)

        glEnable(GL_BLEND)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        transMat = glm.translate(self.pos)
        rotMat = glm.rotate(glm.radians(self.headingAngle), glm.vec3(0.0, 0.0, 1.0))

        self.modelMat = transMat * rotMat

        shader = gSceneManager.GetShader()
        shader.SetMat4('modelMat', self.modelMat)

        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        if self.immortal == True:
            glLineWidth(3.0)
        else:
            glLineWidth(1.0)

        glBindVertexArray(self.drawingStuffVAO)
        glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)

        glBindVertexArray(0)

        glPopAttrib()

    def _Initialize(self):
        verticesData = [
            -self.radius, -self.radius, 2.0, 1.0, 1.0, 1.0, 1.0,
            self.radius, -self.radius, 2.0, 1.0, 1.0, 1.0, 1.0,
            0.0, self.radius, 2.0, 1.0, 1.0, 1.0, 1.0
            ]

        indicesData = [
            0, 1, 2
            ]

        self.vertices = np.array(verticesData, dtype = np.float32)
        self.indices = np.array(indicesData, dtype = np.uint32)

        boardCenter = ((self.boardPos[0][0] + self.boardPos[1][0]) / 2, (self.boardPos[0][1] + self.boardPos[1][1]) / 2)
        self.pos = glm.vec3(boardCenter[0], boardCenter[1], 0.0)

        glBindVertexArray(self.drawingStuffVAO)

        glBindBuffer(GL_ARRAY_BUFFER, self.drawingStuffVBO)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.drawingStuffEBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.vertices.itemsize * self.numVertexComponents, ctypes.c_void_p(0))

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, self.vertices.itemsize * self.numVertexComponents, ctypes.c_void_p(self.vertices.itemsize * 3))

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def _CheckOffScreen(self):
        boardWidth = self.boardPos[1][0] - self.boardPos[0][0]
        boardHeight = self.boardPos[1][1] - self.boardPos[0][1]

        if self.pos.x > self.boardPos[0][0] + boardWidth + self.radius:
            self.pos.x = -self.radius
        elif self.pos.x < -self.radius:
            self.pos.x = self.boardPos[0][0] + boardWidth + self.radius

        if self.pos.y > self.boardPos[0][1] + boardHeight + self.radius:
            self.pos.y = -self.radius
        elif self.pos.y < -self.radius:
            self.pos.y = self.boardPos[0][1] + boardHeight + self.radius

class Asteroid:
    def __init__(self, boardPos, numEnemyColors, pos = None, radius = 0, colorIndex = -1):
        self.vertices = []
        self.indices = []

        self.vel = glm.vec3()

        if pos != None:
            self.pos = glm.vec3(pos)
        else:
            self.pos = None

        self.rotAngle = 0.0
        self.rotAngleInterval = random.randrange(-100, 101)

        self.modelMat = glm.mat4()

        self.numPoints = random.randrange(5, 16)

        if radius != 0:
            self.radius = radius * 0.5
        else:
            self.radius = random.randrange(15, 51)

        if colorIndex != -1:
            self.colorIndex = colorIndex
        else:
            self.colorIndex = random.randrange(0, numEnemyColors)

        self.boardPos = boardPos        

        self.numEnemyColors = numEnemyColors
        self.numVertexComponents = 7
        self.numDrawingStuff = 1

        self.drawingStuffVAO = glGenVertexArrays(self.numDrawingStuff)
        self.drawingStuffVBO = glGenBuffers(self.numDrawingStuff)
        self.drawingStuffEBO = glGenBuffers(self.numDrawingStuff)

        self._Initialize()

    def GetPos(self):
        return self.pos

    def GetRadius(self):
        return self.radius

    def GetColorIndex(self):
        return self.colorIndex

    def BreakUp(self):
        newAsteroids = []
        newAsteroids.append(Asteroid(self.boardPos, self.numEnemyColors, self.pos, self.radius, self.colorIndex))
        newAsteroids.append(Asteroid(self.boardPos, self.numEnemyColors, self.pos, self.radius, self.colorIndex))

        return newAsteroids

    def Update(self, deltaTime):
        self.pos += self.vel * deltaTime

        self.rotAngle += self.rotAngleInterval * deltaTime

        if self.rotAngle > 360.0:
            self.rotAngle -= 360.0

        self._CheckOffScreen()

    def Draw(self):
        glPushAttrib(GL_COLOR_BUFFER_BIT | GL_ENABLE_BIT | GL_POLYGON_BIT)

        glEnable(GL_BLEND)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        transMat = glm.translate(self.pos)
        rotMat = glm.rotate(glm.radians(self.rotAngle), glm.vec3(0.0, 0.0, 1.0))

        self.modelMat = transMat * rotMat

        shader = gSceneManager.GetShader()
        shader.SetMat4('modelMat', self.modelMat)

        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        glBindVertexArray(self.drawingStuffVAO)
        glDrawElements(GL_POLYGON, len(self.indices), GL_UNSIGNED_INT, None)

        glBindVertexArray(0)

        glPopAttrib()

    def _GeneratePolygonData(self, numCirclePoints, xRadius, yRadius, polygonPointsOffset):
        polygonVerticesData = []
        polygonIndicesData = []

        deltaAngle = 360.0 / numCirclePoints

        for i in range(numCirclePoints):
            offsetX = xRadius + polygonPointsOffset[i]
            offsetY = yRadius + polygonPointsOffset[i]

            polygonVerticesData.append(math.cos(glm.radians(i * deltaAngle)) * offsetX)
            polygonVerticesData.append(math.sin(glm.radians(i * deltaAngle)) * offsetY)
            polygonVerticesData.append(0.0)

            color = gSceneManager.GetColor('ObjectColor_', self.colorIndex)

            polygonVerticesData.append(color[0])
            polygonVerticesData.append(color[1])
            polygonVerticesData.append(color[2])
            polygonVerticesData.append(1.0)

            polygonIndicesData.append(i)

        return polygonVerticesData, polygonIndicesData

    def _Initialize(self):
        velX = random.random() - 0.5
        velY = random.random() - 0.5

        self.vel = glm.vec3(velX, velY, 0.0)
        self.vel = glm.normalize(self.vel)
        self.vel *= 50

        polygonPointsOffset = []

        polyOffsetRandRange = math.floor(self.radius * 0.5)

        for i in range(self.numPoints):
            polygonPointsOffset.append(random.randrange(-polyOffsetRandRange, polyOffsetRandRange + 1))

        polygonVerticesData, polygonIndicesData = self._GeneratePolygonData(self.numPoints, self.radius, self.radius, polygonPointsOffset)

        self.vertices = np.array(polygonVerticesData, dtype = np.float32)
        self.indices = np.array(polygonIndicesData, dtype = np.uint32)

        if self.pos == None:
            posX = random.random()
            posY = random.random()

            boardWidth = self.boardPos[1][0] - self.boardPos[0][0]
            boardHeight = self.boardPos[1][1] - self.boardPos[0][1]

            posX = posX * boardWidth + self.boardPos[0][0]
            posY = posY * boardHeight + self.boardPos[0][1]

            self.pos = glm.vec3(posX, posY, 0.0)

        glBindVertexArray(self.drawingStuffVAO)

        glBindBuffer(GL_ARRAY_BUFFER, self.drawingStuffVBO)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.drawingStuffEBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.vertices.itemsize * self.numVertexComponents, ctypes.c_void_p(0))

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, self.vertices.itemsize * self.numVertexComponents, ctypes.c_void_p(self.vertices.itemsize * 3))

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def _CheckOffScreen(self):
        boardWidth = self.boardPos[1][0] - self.boardPos[0][0]
        boardHeight = self.boardPos[1][1] - self.boardPos[0][1]

        if self.pos.x > self.boardPos[0][0] + boardWidth + self.radius:
            self.pos.x = -self.radius
        elif self.pos.x < -self.radius:
            self.pos.x = self.boardPos[0][0] + boardWidth + self.radius

        if self.pos.y > self.boardPos[0][1] + boardHeight + self.radius:
            self.pos.y = -self.radius
        elif self.pos.y < -self.radius:
            self.pos.y = self.boardPos[0][1] + boardHeight + self.radius

class Laser:
    def __init__(self, boardPos, shipPos, shipHeadingVec, shipRadius):
        self.vertices = []

        self.acc = shipHeadingVec * 50000
        self.vel = glm.vec3()
        self.pos = glm.vec3(shipPos.x + shipHeadingVec.x * shipRadius, shipPos.y + shipHeadingVec.y * shipRadius, 0.0)

        self.modelMAt = glm.mat4()

        self.dirty = True

        self.boardPos = boardPos

        self.numVertexComponents = 7
        self.numDrawingStuff = 1

        self.drawingStuffVAO = glGenVertexArrays(self.numDrawingStuff)
        self.drawingStuffVBO = glGenBuffers(self.numDrawingStuff)

        self._Initialize()

    def CheckOffScreen(self):
        boardWidth = self.boardPos[1][0] - self.boardPos[0][0]
        boardHeight = self.boardPos[1][1] - self.boardPos[0][1]

        if self.pos.x > self.boardPos[0][0] + boardWidth or self.pos.x < self.boardPos[0][0]:
            return True

        if self.pos.y > self.boardPos[0][1] + boardHeight or self.pos.y < self.boardPos[0][1]:
            return True

    def CheckHit(self, asteroid):
        dis = glm.length(self.pos - asteroid.GetPos())

        if dis < asteroid.GetRadius():
            return True

        return False

    def Update(self, deltaTime):
        if self.dirty == True:
            self.vel += self.acc * deltaTime

            self.dirty = False

        self.pos += self.vel * deltaTime

    def Draw(self):
        glPushAttrib(GL_COLOR_BUFFER_BIT | GL_ENABLE_BIT | GL_POINT_BIT)

        glEnable(GL_POINT_SMOOTH)
        glEnable(GL_BLEND)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glPointSize(5.0)

        transMat = glm.translate(self.pos)

        self.modelMat = transMat

        shader = gSceneManager.GetShader()
        shader.SetMat4('modelMat', self.modelMat)

        glBindVertexArray(self.drawingStuffVAO)
        glDrawArrays(GL_POINTS, 0, 1)

        glBindVertexArray(0)

        glPopAttrib()

    def _Initialize(self):
        verticesData = [
            0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0
            ]

        self.vertices = np.array(verticesData, dtype = np.float32)

        glBindVertexArray(self.drawingStuffVAO)

        glBindBuffer(GL_ARRAY_BUFFER, self.drawingStuffVBO)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.vertices.itemsize * self.numVertexComponents, ctypes.c_void_p(0))

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, self.vertices.itemsize * self.numVertexComponents, ctypes.c_void_p(self.vertices.itemsize * 3))

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

class Asteroids:
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

        self.gameStuffVerticesList = []
        self.gameStuffIndicesList =[]

        self.removedAsteroidsDistributionVertices = []
        self.removedAsteroidsDistributionIndices = []

        self.backgroundModelMat = glm.mat4()
        self.gameModelMat = glm.mat4()

        self.generateAsteroidsElapsedTime = 0.0
        self.generateAsteroidsTimeLimit = 10.0
        self.numGenerateAsteroids = 5

        self.boardSize = (560, 560)
        self.boardPos = []

        self.displaySize = (displaySize[0], displaySize[1])

        self.ship = None
        self.asteroids = []
        self.lasers = []

        self.removedAsteroidsDistribution = []

        self.numEnemyColors = 7
        self.numVerticesOneBlock = 4
        self.numVertexComponents = 7
        self.numBackgroundStuff = 4
        self.numGameStuff = 1

        self.backgroundVAO = glGenVertexArrays(self.numBackgroundStuff)
        self.backgroundVBO = glGenBuffers(self.numBackgroundStuff)
        self.backgroundEBO = glGenBuffers(self.numBackgroundStuff)

        self.gameVAO = glGenVertexArrays(self.numGameStuff)
        self.gameVBO = glGenBuffers(self.numGameStuff)
        self.gameEBO = glGenBuffers(self.numGameStuff)

        self._InitializeBackgroundStuff()

        self._InitializeGameStuff()

    def Restart(self):
        self.generateAsteroidsElapsedTime = 0.0

        self._InitializeGameStuff()

    def UpdateAboutKeyInput(self, key, value = True):
        if key == glfw.KEY_LEFT or key == glfw.KEY_RIGHT or key == glfw.KEY_UP:
            self.ship.UpdateAboutKeyInput(key, value)

        if key == glfw.KEY_SPACE:
            self.lasers.append(Laser(self.boardPos, self.ship.GetPos(), self.ship.GetHeadingVec(), self.ship.GetRadius()))

    def UpdateAboutMouseInput(self, button, pos):
        if gSceneManager.GetScreenControlWithMouse() == False:
            return

    def Update(self, deltaTime):
        self.ship.Update(deltaTime)

        for i in range(len(self.asteroids)):
            self.asteroids[i].Update(deltaTime)
        
            if self.ship.CheckHit(self.asteroids[i]):
                self.Restart()
                return

        for i in range(len(self.lasers) - 1, -1, -1):
            self.lasers[i].Update(deltaTime)

            if self.lasers[i].CheckOffScreen():
                self.lasers.pop(i)        
            else:
                for j in range(len(self.asteroids) - 1, -1, -1):
                    if self.lasers[i].CheckHit(self.asteroids[j]):
                        if self.asteroids[j].GetRadius() > 10:
                            newAsteroids = self.asteroids[j].BreakUp()
                            self.asteroids.extend(newAsteroids)
                        else:
                            colorIndex = self.asteroids[j].GetColorIndex()
                            self.removedAsteroidsDistribution[colorIndex] += 1

                        self.asteroids.pop(j)
                        self.lasers.pop(i)

                        break

        self.generateAsteroidsElapsedTime += deltaTime

        if self.generateAsteroidsElapsedTime > self.generateAsteroidsTimeLimit:
            self.generateAsteroidsElapsedTime = 0.0

            for i in range(self.numGenerateAsteroids):
                self.asteroids.append(Asteroid(self.boardPos, self.numEnemyColors))

            self.ship.SetImmortal(True)

    def Draw(self):
        self._DrawBackground()

        for i in range(len(self.asteroids)):
            self.asteroids[i].Draw()

        for i in range(len(self.lasers)):
            self.lasers[i].Draw()

        self.ship.Draw()

        self._DrawRemovedAsteroidsDistribution()

    def _InitializeBackgroundStuff(self):
        boardLbPos = [20.0, 20.0]
        boardRtPos = []
        boardRtPos.append(boardLbPos[0] + self.boardSize[0])
        boardRtPos.append(boardLbPos[1] + self.boardSize[1])

        self.boardPos.append(boardLbPos)
        self.boardPos.append(boardRtPos)

        backgroundVerticesData = [
            0.0, 0.0, 5.0, 1.0, 0.0, 0.0, 0.5,
            self.boardPos[0][0], 0.0, 5.0, 1.0, 0.0, 0.0, 0.5,
            self.boardPos[0][0], self.displaySize[1], 5.0, 1.0, 0.0, 0.0, 0.5,
            0.0, self.displaySize[1], 5.0, 1.0, 0.0, 0.0, 0.5,

            self.boardPos[1][0], 0.0, 5.0, 1.0, 0.0, 0.0, 0.5,
            self.displaySize[0], 0.0, 5.0, 1.0, 0.0, 0.0, 0.5,
            self.displaySize[0], self.displaySize[1], 5.0, 1.0, 0.0, 0.0, 0.5,
            self.boardPos[1][0], self.displaySize[1], 5.0, 1.0, 0.0, 0.0, 0.5,

            0.0, 0.0, 5.0, 1.0, 0.0, 0.0, 0.5,
            self.displaySize[0], 0.0, 5.0, 1.0, 0.0, 0.0, 0.5,
            self.displaySize[0], self.boardPos[0][1], 5.0, 1.0, 0.0, 0.0, 0.5,
            0.0, self.boardPos[0][1], 5.0, 1.0, 0.0, 0.0, 0.5,

            0.0, self.boardPos[1][1], 5.0, 1.0, 0.0, 0.0, 0.5,
            self.displaySize[0], self.boardPos[1][1], 5.0, 1.0, 0.0, 0.0, 0.5,
            self.displaySize[0], self.displaySize[1], 5.0, 1.0, 0.0, 0.0, 0.5,
            0.0, self.displaySize[1], 5.0, 1.0, 0.0, 0.0, 0.5
            ]

        backgroundIndicesData = [
            0, 1, 2, 2, 3, 0,
            4, 5, 6, 6, 7 ,4,
            8, 9, 10, 10, 11, 8,
            12, 13, 14, 14, 15, 12
            ]

        self.backgroundVertices = np.array(backgroundVerticesData, dtype = np.float32)
        self.backgroundIndices = np.array(backgroundIndicesData, dtype = np.uint32)

        backgroundSubAreaVerticesData = [
            600.0, 40.0, 7.0, 0.0, 0.0, 0.0, 1.0,
            780.0, 40.0, 7.0, 0.0, 0.0, 0.0, 1.0,
            780.0, 240.0, 7.0, 0.0, 0.0, 0.0, 1.0,
            600.0, 240.0, 7.0, 0.0, 0.0, 0.0, 1.0
            ]

        backgroundSubAreaIndicesData = [
            0, 1, 2, 2, 3, 0
            ]

        self.backgroundSubAreaVertices = np.array(backgroundSubAreaVerticesData, dtype = np.float32)
        self.backgroundSubAreaIndices = np.array(backgroundSubAreaIndicesData, dtype = np.uint32)

        backgroundLineVerticesData = [
            0.0, self.boardPos[0][1], 8.0, 1.0, 1.0, 1.0, 1.0,
            self.displaySize[0], self.boardPos[0][1], 8.0, 1.0, 1.0, 1.0, 1.0,

            0.0, self.boardPos[1][1], 8.0, 1.0, 1.0, 1.0, 1.0,
            self.displaySize[0], self.boardPos[1][1], 8.0, 1.0, 1.0, 1.0, 1.0,

            self.boardPos[0][0], 0.0, 8.0, 1.0, 1.0, 1.0, 1.0,
            self.boardPos[0][0], self.displaySize[1], 8.0, 1.0, 1.0, 1.0, 1.0,

            self.boardPos[1][0], 0.0, 8.0, 1.0, 1.0, 1.0, 1.0,
            self.boardPos[1][0], self.displaySize[1], 8.0, 1.0, 1.0, 1.0, 1.0
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
            backgroundSubAreaVerticesData[0], backgroundSubAreaVerticesData[1], 8.0, 1.0, 1.0, 1.0, 1.0,
            backgroundSubAreaVerticesData[7], backgroundSubAreaVerticesData[8], 8.0, 1.0, 1.0, 1.0, 1.0,
            backgroundSubAreaVerticesData[14], backgroundSubAreaVerticesData[15], 8.0, 1.0, 1.0, 1.0, 1.0,
            backgroundSubAreaVerticesData[21], backgroundSubAreaVerticesData[22], 8.0, 1.0, 1.0, 1.0, 1.0
            ]

        backgroundSubAreaLineIndicesData = [
            0, 1,
            1, 2,
            2, 3,
            3, 0
            ]

        self.backgroundSubAreaLineVertices = np.array(backgroundSubAreaLineVerticesData, dtype = np.float32)
        self.backgroundSubAreaLineIndices = np.array(backgroundSubAreaLineIndicesData, dtype = np.uint32)

        self.backgroundStuffVerticesList.clear()
        self.backgroundStuffIndicesList.clear()

        self.backgroundStuffVerticesList.append(self.backgroundVertices)
        self.backgroundStuffVerticesList.append(self.backgroundSubAreaVertices)
        self.backgroundStuffVerticesList.append(self.backgroundLineVertices)
        self.backgroundStuffVerticesList.append(self.backgroundSubAreaLineVertices)

        self.backgroundStuffIndicesList.append(self.backgroundIndices)
        self.backgroundStuffIndicesList.append(self.backgroundSubAreaIndices)
        self.backgroundStuffIndicesList.append(self.backgroundLineIndices)
        self.backgroundStuffIndicesList.append(self.backgroundSubAreaLineIndices)

        for i in range(self.numBackgroundStuff):
            glBindVertexArray(self.backgroundVAO[i])

            glBindBuffer(GL_ARRAY_BUFFER, self.backgroundVBO[i])
            glBufferData(GL_ARRAY_BUFFER, self.backgroundStuffVerticesList[i].nbytes, self.backgroundStuffVerticesList[i], GL_STATIC_DRAW)

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.backgroundEBO[i])
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.backgroundStuffIndicesList[i].nbytes, self.backgroundStuffIndicesList[i], GL_STATIC_DRAW)

            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.backgroundStuffVerticesList[i].itemsize * self.numVertexComponents, ctypes.c_void_p(0))

            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, self.backgroundStuffVerticesList[i].itemsize * self.numVertexComponents, ctypes.c_void_p(self.backgroundStuffVerticesList[i].itemsize * 3))

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def _InitializeGameStuff(self):
        del self.ship
        self.asteroids.clear()

        self.ship = Ship(self.boardPos)

        for i in range(self.numGenerateAsteroids):
            self.asteroids.append(Asteroid(self.boardPos, self.numEnemyColors))

        removedAsteroidsDistributionVerticesData = []
        removedAsteroidsDistributionIndicesData = []

        for i in range(self.numEnemyColors):
            for j in range(self.numVerticesOneBlock):
                removedAsteroidsDistributionVerticesData.append(0.0)
                removedAsteroidsDistributionVerticesData.append(0.0)
                removedAsteroidsDistributionVerticesData.append(7.5)

                color = gSceneManager.GetColor('ObjectColor_', i)

                removedAsteroidsDistributionVerticesData.append(color[0])
                removedAsteroidsDistributionVerticesData.append(color[1])
                removedAsteroidsDistributionVerticesData.append(color[2])
                removedAsteroidsDistributionVerticesData.append(0.8)

        for i in range(self.numEnemyColors):
            iOffset = i * self.numVerticesOneBlock

            removedAsteroidsDistributionIndicesData.append(iOffset + 0)
            removedAsteroidsDistributionIndicesData.append(iOffset + 1)
            removedAsteroidsDistributionIndicesData.append(iOffset + 2)

            removedAsteroidsDistributionIndicesData.append(iOffset + 2)
            removedAsteroidsDistributionIndicesData.append(iOffset + 3)
            removedAsteroidsDistributionIndicesData.append(iOffset + 0)

        self.removedAsteroidsDistributionVertices = np.array(removedAsteroidsDistributionVerticesData, dtype = np.float32)
        self.removedAsteroidsDistributionIndices = np.array(removedAsteroidsDistributionIndicesData, dtype = np.uint32)

        self.gameStuffVerticesList.clear()
        self.gameStuffIndicesList.clear()

        self.gameStuffVerticesList.append(self.removedAsteroidsDistributionVertices)
        self.gameStuffIndicesList.append(self.removedAsteroidsDistributionIndices)

        glBindVertexArray(self.gameVAO)

        glBindBuffer(GL_ARRAY_BUFFER, self.gameVBO)
        glBufferData(GL_ARRAY_BUFFER, self.gameStuffVerticesList[0].nbytes, self.gameStuffVerticesList[0], GL_DYNAMIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.gameEBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.gameStuffIndicesList[0].nbytes, self.gameStuffIndicesList[0], GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.gameStuffVerticesList[0].itemsize * self.numVertexComponents, ctypes.c_void_p(0))

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, self.gameStuffVerticesList[0].itemsize * self.numVertexComponents, ctypes.c_void_p(self.gameStuffVerticesList[0].itemsize * 3))

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        self.removedAsteroidsDistribution = [0 for i in range(self.numEnemyColors)]

    def _DrawBackground(self):
        glPushAttrib(GL_COLOR_BUFFER_BIT | GL_ENABLE_BIT | GL_LINE_BIT)

        glEnable(GL_BLEND)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        shader = gSceneManager.GetShader()
        shader.SetMat4('modelMat', self.backgroundModelMat)

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

        glBindVertexArray(0)

        glPopAttrib()

    def _DrawRemovedAsteroidsDistribution(self):
        glPushAttrib(GL_COLOR_BUFFER_BIT | GL_ENABLE_BIT)

        glEnable(GL_BLEND)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        shader = gSceneManager.GetShader()
        shader.SetMat4('modelMat', self.gameModelMat)

        distributionBlockWidth = 20
        distributionBlockYInterval = 3

        posX = (self.backgroundSubAreaVertices[7] - self.backgroundSubAreaVertices[0]) - (self.numEnemyColors * distributionBlockWidth)
        posX = self.backgroundSubAreaVertices[0] + posX / 2.0
        posY = self.backgroundSubAreaVertices[1] + 20

        for i in range(self.numEnemyColors):
            iOffset = i * self.numVerticesOneBlock * self.numVertexComponents

            posX += distributionBlockWidth

            verticesIndex = 0
            vOffset = verticesIndex * self.numVertexComponents

            self.removedAsteroidsDistributionVertices[iOffset + vOffset + 0] = posX - distributionBlockWidth
            self.removedAsteroidsDistributionVertices[iOffset + vOffset + 1] = posY

            verticesIndex += 1
            vOffset = verticesIndex * self.numVertexComponents

            self.removedAsteroidsDistributionVertices[iOffset + vOffset + 0] = posX
            self.removedAsteroidsDistributionVertices[iOffset + vOffset + 1] = posY

            verticesIndex += 1
            vOffset = verticesIndex * self.numVertexComponents

            self.removedAsteroidsDistributionVertices[iOffset + vOffset + 0] = posX
            self.removedAsteroidsDistributionVertices[iOffset + vOffset + 1] = posY + self.removedAsteroidsDistribution[i] * distributionBlockYInterval

            verticesIndex += 1
            vOffset = verticesIndex * self.numVertexComponents

            self.removedAsteroidsDistributionVertices[iOffset + vOffset + 0] = posX - distributionBlockWidth
            self.removedAsteroidsDistributionVertices[iOffset + vOffset + 1] = posY + self.removedAsteroidsDistribution[i] * distributionBlockYInterval

        glBindBuffer(GL_ARRAY_BUFFER, self.gameVBO)
        glBufferSubData(GL_ARRAY_BUFFER, 0, self.gameStuffVerticesList[0].nbytes, self.gameStuffVerticesList[0])

        glBindVertexArray(self.gameVAO)
        glDrawElements(GL_TRIANGLES, len(self.gameStuffIndicesList[0]), GL_UNSIGNED_INT, None)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
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
        if key == glfw.KEY_E:
            gInputManager.SetKeyState('E', True)
        if key == glfw.KEY_G:
            gInputManager.SetKeyState('G', True)
        if key == glfw.KEY_I:
            gInputManager.SetKeyState('I', True) 
        if key == glfw.KEY_P:
            gInputManager.SetKeyState('P', True)            
        if key == glfw.KEY_R:
            gInputManager.SetKeyState('R', True)
        if key == glfw.KEY_S:
            gInputManager.SetKeyState('S', True)

        if key == glfw.KEY_W:
            gInputManager.SetKeyState('W', True)
        elif key == glfw.KEY_S:
            gInputManager.SetKeyState('S', True)
        elif key == glfw.KEY_A:
            gInputManager.SetKeyState('A', True)
        elif key == glfw.KEY_D:
            gInputManager.SetKeyState('D', True)

        if key == glfw.KEY_DOWN:
            gInputManager.SetKeyState(glfw.KEY_DOWN, True)

        if key == glfw.KEY_LEFT:
            gInputManager.SetKeyState(glfw.KEY_LEFT, True)
        elif key == glfw.KEY_RIGHT:
            gInputManager.SetKeyState(glfw.KEY_RIGHT, True)
        elif key == glfw.KEY_UP:
            gInputManager.SetKeyState(glfw.KEY_UP, True)

    if action == glfw.RELEASE:
        if key == glfw.KEY_SPACE:
            gInputManager.SetKeyState(glfw.KEY_SPACE, False)

        if key == glfw.KEY_W:
            gInputManager.SetKeyState('W', False)
        elif key == glfw.KEY_S:
            gInputManager.SetKeyState('S', False)
        elif key == glfw.KEY_A:
            gInputManager.SetKeyState('A', False)
        elif key == glfw.KEY_D:
            gInputManager.SetKeyState('D', False)

        if key == glfw.KEY_LEFT:
            gInputManager.SetKeyState(glfw.KEY_LEFT, False)
        elif key == glfw.KEY_RIGHT:
            gInputManager.SetKeyState(glfw.KEY_RIGHT, False)
        elif key == glfw.KEY_UP:
            gInputManager.SetKeyState(glfw.KEY_UP, False)

def HandleMouseButtonCallback(glfwWindow, button, action, mod):
    if button == glfw.MOUSE_BUTTON_LEFT:
        if action == glfw.PRESS:            
            gInputManager.SetMouseButtonClick(glfw.MOUSE_BUTTON_LEFT, True)
            gInputManager.SetLastMousePosOnClick(glfw.get_cursor_pos(glfwWindow))
        elif action == glfw.RELEASE:            
            gInputManager.SetMouseButtonClick(glfw.MOUSE_BUTTON_LEFT, False)
            gInputManager.SetLastMousePosOnClick(glfw.get_cursor_pos(glfwWindow))

def HandleCursorPosCallback(glfwWindow, xPos, yPos):
    if gSceneManager.GetPause() == True:
        return

    if gSceneManager.GetScreenControlWithMouse() == False:
        return

    if gSceneManager.GetAbleMouseDragged() == True:
        if gInputManager.GetMouseButtonClick(glfw.MOUSE_BUTTON_LEFT) == True:
            gInputManager.SetLastMousePosOnClick([xPos, yPos])

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

    if gSceneManager.GetAbleMouseEscape() == False:
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

    glfwWindow = glfw.create_window(displaySize[0], displaySize[1], 'Asteroids.Part 1', None, None)

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

    gSceneManager.InitializeOpenGL(shader)
    gSceneManager.SetCamera(Camera())
    gSceneManager.MakeFont()    
    gSceneManager.AddObject(Asteroids(displaySize))
    
    lastElapsedTime = glfw.get_time()
    deltaTime = 0.0

    while glfw.window_should_close(glfwWindow) == False:
        glfw.poll_events()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        gSceneManager.Update(deltaTime)

        gSceneManager.DrawObjects()        

        gSceneManager.DrawProgramInfo(deltaTime)

        glfw.swap_buffers(glfwWindow)

        deltaTime = glfw.get_time() - lastElapsedTime
        lastElapsedTime = glfw.get_time()


glfw.terminate()


if __name__ == "__main__":
    Main()