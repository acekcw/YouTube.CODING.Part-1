
from OpenGL.GL import *
from OpenGL.GL.shaders import *

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


class Index:
    SPECIFIC_PROGRAM_INFO_COLOR = 0
    SPECIFIC_PROGRAM_INFO_TEXT_POS = 1
    SPECIFIC_PROGRAM_INFO_TEXT_INTERVAL = 2
    SPECIFIC_PROGRAM_INFO_NUM_TEXTS = 3
    SPECIFIC_PROGRAM_INFO_FONT_SIZE = 4
    SPECIFIC_PROGRAM_INFO_TEXT_0 = 5   

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

        self.view3D = view3D

        self.perspectivePrjMat = glm.perspective(self.fovy, self.aspect, self.near, self.far)
        self.orthoPrjMat = glm.ortho(0, self.displaySize[0], 0, self.displaySize[1], 1.0, 100.0)

        self.view3DMat = glm.mat4()
        self.view2DMat = glm.translate(glm.vec3(0.0, 0.0, 20.0))
        self.view2DMat = np.linalg.inv(self.view2DMat)

        self.programInfoAreaModelMat = glm.mat4() 

        self.screenControlWithMouse = False
        self.ableMouseDragged = False
        self.ableMouseEscape = True

        self.objects = []
        self.font = None
        self.largeFont = None

        self.deltaTime = 0.0
        self.dirty = True

        self.colors = {}

        self.programInfo = False
        self.numProgramInfoElement = 8

        self.specificProgramInfo = True
        self.specificProgramArgs = []

        self.controlFPS = False
        self.FPS = 30
        self.oneFrameTime = 1.0 / self.FPS
        self.elapsedTime = 0.0        
        self.enableRender = True

        self.pause = False
        self.debug = False
        self.debugMat = glm.mat4()

        self.numVertexComponents = 7
        self.numDrawingStuff = 1

        self.drawingStuffVAO = None
        self.drawingStuffVBO = None
        self.drawingStuffEBO = None        

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
        if gInputManager.GetKeyState(glfw.KEY_W) == True:
            self.camera.ProcessKeyboard('FORWARD', 0.05)
            self.dirty = True
        if gInputManager.GetKeyState(glfw.KEY_S) == True:
            self.camera.ProcessKeyboard('BACKWARD', 0.05)
            self.dirty = True
        if gInputManager.GetKeyState(glfw.KEY_A) == True:
            self.camera.ProcessKeyboard('LEFT', 0.05)
            self.dirty = True
        if gInputManager.GetKeyState(glfw.KEY_D) == True:
            self.camera.ProcessKeyboard('RIGHT', 0.05)
            self.dirty = True 

    def SailCamera(self):
        if self.sailingCamera[0] == True:
            self.camera.ProcessKeyboard('FORWARD', 1.0)
            self.dirty = True
        if self.sailingCamera[1] == True:
            self.camera.ProcessKeyboard('BACKWARD', 1.0)
            self.dirty = True

    def SetSpecificProgramArgs(self, index, subIndex, value):        
        argsList = list(self.specificProgramArgs[index])

        argsList[subIndex] = value     

        self.specificProgramArgs[index] = tuple(argsList)

    def InitializeOpenGL(self, shader):        
        self.shader = shader        

        color = self.GetColor('DefaultColor_', 1)
        glClearColor(color[0], color[1], color[2], 1.0)

        glEnable(GL_DEPTH_TEST)

        self.drawingStuffVAO = glGenVertexArrays(self.numDrawingStuff)
        self.drawingStuffVBO = glGenBuffers(self.numDrawingStuff)
        self.drawingStuffEBO = glGenBuffers(self.numDrawingStuff)

        glBindVertexArray(self.drawingStuffVAO)

        glBindBuffer(GL_ARRAY_BUFFER, self.drawingStuffVBO)
        glBufferData(GL_ARRAY_BUFFER, self.programInfoAreaVertices.nbytes, self.programInfoAreaVertices, GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.drawingStuffEBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.programInfoAreaIndices.nbytes, self.programInfoAreaIndices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.programInfoAreaVertices.itemsize * self.numVertexComponents, ctypes.c_void_p(0))

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, self.programInfoAreaVertices.itemsize * self.numVertexComponents, ctypes.c_void_p(self.programInfoAreaVertices.itemsize * 3))

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def MakeFont(self):
        self.font = Font('..\Resources\Fonts\comic.ttf', 14)
        self.largeFont = Font('..\Resources\Fonts\comic.ttf', 21)

        #self.font = Font('..\Resources\Fonts\cour.ttf', 14)
        #self.font = Font('..\Resources\Fonts\HYNAMM.ttf', 14)

        self.font.MakeFontTextureWithGenList()
        self.largeFont.MakeFontTextureWithGenList()

    def AddObject(self, object):
        self.objects.append(object)

    def AddSpecificProgramArgs(self, *args):
        self.specificProgramArgs.append(args)

    def ClearSpecificProgramArgs(self):
        self.specificProgramArgs.clear()

    def UpdateAboutKeyInput(self):
        numObjects = len(self.objects)

        if gInputManager.GetKeyState(glfw.KEY_SPACE) == True:
            for i in range(numObjects):
                self.objects[i].UpdateAboutKeyInput(glfw.KEY_SPACE)
            gInputManager.SetKeyState(glfw.KEY_SPACE, False)    

        if gInputManager.GetKeyState(glfw.KEY_1) == True:
            self.sailingCamera[0] = not self.sailingCamera[0]
            gInputManager.SetKeyState(glfw.KEY_1, False)        
        if gInputManager.GetKeyState(glfw.KEY_2) == True:
            self.sailingCamera[1] = not self.sailingCamera[1]
            gInputManager.SetKeyState(glfw.KEY_2, False)
        if gInputManager.GetKeyState(glfw.KEY_3) == True:
            for i in range(numObjects):
                self.objects[i].UpdateAboutKeyInput(glfw.KEY_3)
            gInputManager.SetKeyState(glfw.KEY_3, False)
        if gInputManager.GetKeyState(glfw.KEY_4) == True:
            for i in range(numObjects):
                self.objects[i].UpdateAboutKeyInput(glfw.KEY_4)
            gInputManager.SetKeyState(glfw.KEY_4, False)

        if gInputManager.GetKeyState(glfw.KEY_B) == True:
            self.debug = not self.debug
            gInputManager.SetKeyState(glfw.KEY_B, False)
        if gInputManager.GetKeyState(glfw.KEY_C) == True:
            self.screenControlWithMouse = not self.screenControlWithMouse            
            gInputManager.SetKeyState(glfw.KEY_C, False)
        if gInputManager.GetKeyState(glfw.KEY_E) == True:
            self.ableMouseEscape = not self.ableMouseEscape
            gInputManager.SetKeyState(glfw.KEY_E, False)
        if gInputManager.GetKeyState(glfw.KEY_F) == True:
            self.specificProgramInfo = not self.specificProgramInfo
            gInputManager.SetKeyState(glfw.KEY_F, False)
        if gInputManager.GetKeyState(glfw.KEY_G) == True:
            self.ableMouseDragged = not self.ableMouseDragged
            gInputManager.SetKeyState(glfw.KEY_G, False)
        if gInputManager.GetKeyState(glfw.KEY_I) == True:
            self.programInfo = not self.programInfo                
            gInputManager.SetKeyState(glfw.KEY_I, False)
        if gInputManager.GetKeyState(glfw.KEY_P) == True:
            self.pause = not self.pause            
            gInputManager.SetKeyState(glfw.KEY_P, False)
        if gInputManager.GetKeyState(glfw.KEY_R) == True:
            for i in range(numObjects):
                self.objects[i].Restart()
            gInputManager.SetKeyState(glfw.KEY_R, False)
        if gInputManager.GetKeyState(glfw.KEY_V) == True:
            self.view3D = not self.view3D            
            gInputManager.SetKeyState(glfw.KEY_V, False)

        if gInputManager.GetKeyState(glfw.KEY_LEFT) == True:
            for i in range(numObjects):
                self.objects[i].UpdateAboutKeyInput(glfw.KEY_LEFT)
        if gInputManager.GetKeyState(glfw.KEY_RIGHT) == True:
            for i in range(numObjects):
                self.objects[i].UpdateAboutKeyInput(glfw.KEY_RIGHT)  
        if gInputManager.GetKeyState(glfw.KEY_UP) == True:            
            for i in range(numObjects):
                self.objects[i].UpdateAboutKeyInput(glfw.KEY_UP)
        if gInputManager.GetKeyState(glfw.KEY_DOWN) == True:
            for i in range(numObjects):
                self.objects[i].UpdateAboutKeyInput(glfw.KEY_DOWN)            

    def UpdateAboutMouseInput(self):
        numObjects = len(self.objects)       

        if gInputManager.GetMouseButtonClick(glfw.MOUSE_BUTTON_LEFT) == True:            
            lastMousePosOnClick = gInputManager.GetLastMousePosOnClick()
            for i in range(numObjects):
                self.objects[i].UpdateAboutMouseInput(glfw.MOUSE_BUTTON_LEFT, lastMousePosOnClick)
        if gSceneManager.ableMouseDragged == False:
            gInputManager.SetMouseButtonClick(glfw.MOUSE_BUTTON_LEFT, False)

    def PostUpdate(self, deltaTime):
        if gInputManager.GetKeyState(glfw.KEY_8) == True:
            if self.controlFPS == True:
                self.FPS -= 5
                if self.FPS <= 0:
                    self.FPS = 1

                self.oneFrameTime = 1.0 / self.FPS
                self.elapsedTime = 0.0
                self.enableRender = False

            gInputManager.SetKeyState(glfw.KEY_8, False)

        if gInputManager.GetKeyState(glfw.KEY_9) == True:
            if self.controlFPS == True:
                self.FPS = int(self.FPS / 5) * 5 + 5
                if self.FPS > 100:
                    self.FPS = 100

                self.oneFrameTime = 1.0 / self.FPS
                self.elapsedTime = 0.0
                self.enableRender = False        

            gInputManager.SetKeyState(glfw.KEY_9, False)        

        if gInputManager.GetKeyState(glfw.KEY_0) == True:
            self.controlFPS = not self.controlFPS

            if self.controlFPS == True:
                self.elapsedTime = 0.0        
                self.enableRender = False

            gInputManager.SetKeyState(glfw.KEY_0, False)

        if self.enableRender == True:
            self.elapsedTime = 0.0
            self.enableRender = False

    def Update(self, deltaTime):
        if self.controlFPS == True:
            self.elapsedTime += deltaTime

            if self.elapsedTime < self.oneFrameTime:                
                return        
        
        self.enableRender = True

        self.shader.Use()       

        self.UpdateAboutKeyInput()

        self.UpdateAboutMouseInput()

        if self.pause == True:
            return

        numObjects = len(self.objects)

        for i in range(numObjects):
            if self.controlFPS == True:
                self.objects[i].Update(self.elapsedTime)
            else:
                self.objects[i].Update(deltaTime)

        if self.view3D == True:
            self.SetCameraPos()
            self.SailCamera()

        if self.dirty == False:
            return  

        self.view3DMat = self.camera.GetViewMat()

        self.deltaTime += deltaTime
        self.dirty = False
        
    def Draw(self, deltaTime):
        if self.enableRender != True:            
            return self.enableRender        

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self._DrawObjects(deltaTime)

        self._DrawGUI(deltaTime)

        return self.enableRender

    def _DrawObjects(self, deltaTime):        
        numObjects = len(self.objects)

        if self.view3D == True:
            prjMat = self.perspectivePrjMat
            viewMat = self.view3DMat            
        else:
            prjMat = self.orthoPrjMat
            viewMat = self.view2DMat

        self.shader.SetMat4('prjMat', prjMat)
        self.shader.SetMat4('viewMat', viewMat)        

        for i in range(numObjects):
            self.objects[i].Draw()

    def _DrawGUI(self, deltaTime):
        #self.debugMat = glGetFloatv(GL_MODELVIEW_MATRIX)       

        self._DrawProgramInfoArea()        

        glUseProgram(0)

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()

        glOrtho(0, self.displaySize[0], 0, self.displaySize[1], -10.0, 10.0)        

        glMatrixMode(GL_MODELVIEW)

        self._DrawProgramInfo(deltaTime)

        self._DrawSpecificProgramInfo(deltaTime)

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
            585.0, 590.0, 9.0, 1.0, 1.0, 1.0, 1.0,
            785.0, 590.0, 9.0, 1.0, 1.0, 1.0, 1.0,
            585.0, 587.0, 9.0, 1.0, 1.0, 1.0, 1.0,
            785.0, 587.0, 9.0, 1.0, 1.0, 1.0, 1.0,

            580.0, 570.0, 9.0, 0.0, 0.0, 1.0, 0.8,
            580.0, 370.0, 9.0, 0.0, 0.0, 1.0, 0.8,
            577.0, 570.0, 9.0, 0.0, 0.0, 1.0, 0.8,
            577.0, 370.0, 9.0, 0.0, 0.0, 1.0, 0.8,

            585.0, 353.0, 9.0, 0.0, 0.0, 1.0, 0.8,
            785.0, 353.0, 9.0, 0.0, 0.0, 1.0, 0.8,
            585.0, 350.0, 9.0, 0.0, 0.0, 1.0, 0.8,
            785.0, 350.0, 9.0, 0.0, 0.0, 1.0, 0.8,

            790.0, 570.0, 9.0, 0.0, 0.0, 1.0, 0.8,
            790.0, 370.0, 9.0, 0.0, 0.0, 1.0, 0.8,
            793.0, 570.0, 9.0, 0.0, 0.0, 1.0, 0.8,
            793.0, 370.0, 9.0, 0.0, 0.0, 1.0, 0.8
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
        if self.programInfo == False:
            return

        glPushAttrib(GL_COLOR_BUFFER_BIT | GL_ENABLE_BIT)

        glEnable(GL_BLEND)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)        

        self.shader.SetMat4('prjMat', self.orthoPrjMat)
        self.shader.SetMat4('viewMat', self.view2DMat)       
        self.shader.SetMat4('modelMat', self.programInfoAreaModelMat)
                
        glBindVertexArray(self.drawingStuffVAO)
        glDrawElements(GL_LINES, len(self.programInfoAreaIndices), GL_UNSIGNED_INT, None)        

        glBindVertexArray(0)

        glPopAttrib()        
       
    def _DrawProgramInfo(self, deltaTime):
        if self.programInfo == False:
            return

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
        infoFPSText = ".FPS"

        if self.controlFPS == True:
            infoFPSText += ".On(8, 9; 0)"
        else:
            infoFPSText += ".Off(0)"

        infoText.append(infoFPSText + ' : {0: 0.2f}'.format(0.0))
        
        if self.controlFPS == True:
            if self.elapsedTime != 0.0:
                infoText[infoTextIndex] = infoFPSText + " : {0: 0.2f} ({1})".format(1.0 / self.elapsedTime, self.FPS)
                #print('.FPS: {0: 0.2f} ({1})'.format(1.0 / self.elapsedTime, self.FPS))
        else:
            if deltaTime != 0.0:
                infoText[infoTextIndex] = infoFPSText + " : {0: 0.2f}".format(1.0 / deltaTime)
                #print('.FPS: {0: 0.2f}'.format(1.0 / deltaTime))            

        infoText.append('.ViewMode(V) : ')
        infoTextIndex += 1

        if self.view3D == True:
            infoText[infoTextIndex] += "3D"
        else:
            infoText[infoTextIndex] += "2D"

        infoText.append('.SailingDir(1, 2) : ')
        infoTextIndex += 1

        if self.sailingCamera[0] == True:
            infoText[infoTextIndex] += "F"
        if self.sailingCamera[1] == True:
            infoText[infoTextIndex] += "B"

        infoText.append('.Pause(P) : ')
        infoTextIndex += 1
        
        if self.pause == True:
            infoText[infoTextIndex] += "On"
        else:
            infoText[infoTextIndex] += "Off"

        infoText.append('.SCWithMouse(C) : ')
        infoTextIndex += 1
        
        if self.screenControlWithMouse == True:
            infoText[infoTextIndex] += "On"
        else:
            infoText[infoTextIndex] += "Off"

        infoText.append('.  Dragged(G) : ')
        infoTextIndex += 1
        
        if self.ableMouseDragged == True:
            infoText[infoTextIndex] += "On"
        else:
            infoText[infoTextIndex] += "Off"

        infoText.append('.  MouseEscape(E) : ')
        infoTextIndex += 1
        
        if self.ableMouseEscape == True:
            infoText[infoTextIndex] += "On"
        else:
            infoText[infoTextIndex] += "Off"

        infoText.append('.Debug(B) : ')
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
                textPosY -= 65.0
            else:
                textPosY -= 20.0
        
        glPopAttrib()

        glPopMatrix()

    def _DrawSpecificProgramInfo(self, deltaTime):
        if self.specificProgramInfo == False:
            return
        
        glPushMatrix()
        glLoadIdentity()

        glPushAttrib(GL_COLOR_BUFFER_BIT | GL_ENABLE_BIT)

        glDisable(GL_DEPTH_TEST)

        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)        

        color = []
        textPos = [0.0, 0.0]
        textIntervalY = 0.0
        font = None
        infoText = []
        numInfoTexts = 0

        for i in range(len(self.specificProgramArgs)):
            args = self.specificProgramArgs[i]

            color = args[0]
            glColor(color[0], color[1], color[2], 1.0)

            textPos[0] = args[1][0]
            textPos[1] = args[1][1]
            textIntervalY = args[2]
            numInfoTexts = args[3]

            infoText = args[5 : ]

            if 'Large' == args[4]:
                font = self.largeFont
            elif 'Medium' == args[4]:
                font = self.font

            texId = font.GetTexId()
            glBindTexture(GL_TEXTURE_2D, texId)

            for i in range(numInfoTexts):
                glTranslate(textPos[0], textPos[1], 0.0)

                glListBase(font.GetListOffset())
                glCallLists([ord(c) for c in infoText[i]])        

                glPopMatrix()
                glPushMatrix()
                glLoadIdentity()
            
                textPos[1] -= textIntervalY

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

class TestProgram:
    def __init__(self, displaySize, programName):
        self.programName = programName

        self.vertices = []
        self.indices = []

        self.rotDegree = 0.0

        self.modelMat = glm.mat4()

        self.displaySize = (displaySize[0], displaySize[1])
        
        self.numVertexComponents = 9
        self.numDrawingStuff = 1        

        self.drawingStuffVAO = glGenVertexArrays(self.numDrawingStuff)
        self.drawingStuffVBO = glGenBuffers(self.numDrawingStuff)
        self.drawingStuffEBO = glGenBuffers(self.numDrawingStuff)

        self._Initialize()        

    def Restart(self):
        pass

    def UpdateAboutKeyInput(self, key, value = True):
        pass

    def UpdateAboutMouseInput(self, button, pos):
        if gSceneManager.GetScreenControlWithMouse() == False:
            return

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

        glBindVertexArray(self.drawingStuffVAO)
        glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)

        glBindVertexArray(0)

        glPopAttrib()

    def _Initialize(self):
        verticesData = [            
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

        indicesData = [
            0, 1, 2, 2, 3, 0,
            4, 5, 6, 6, 7, 4,
            8, 9, 10, 10, 11, 8,
            12, 13, 14, 14, 15, 12,
            16, 17, 18, 18, 19, 16,
            20, 21, 22, 22, 23, 20
            ]        

        self.vertices = np.array(verticesData, dtype = np.float32)
        self.indices = np.array(indicesData, dtype = np.uint32)

        glBindVertexArray(self.drawingStuffVAO)

        glBindBuffer(GL_ARRAY_BUFFER, self.drawingStuffVBO)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.drawingStuffEBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.vertices.itemsize * self.numVertexComponents, ctypes.c_void_p(0))

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, self.vertices.itemsize * self.numVertexComponents, ctypes.c_void_p(self.vertices.itemsize * 3))

        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, self.vertices.itemsize * self.numVertexComponents, ctypes.c_void_p(self.vertices.itemsize * 7))

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0)        
        
        gSceneManager.AddSpecificProgramArgs(gSceneManager.GetColor('DefaultColor_', 7), [635, 30], 0, 1, 'Large', self.programName)

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

class Cell:
    def __init__(self, pos, size, boardPos):
        self.vertices = []
        self.indices = []

        self.wallVertices = []
        self.wallIndices = []

        self.pos = glm.vec3(pos[0], pos[1], 0.0)
        self.size = size

        self.walls = [True, True, True, True]
        self.neighbors = []

        self.visited = False

        self.numDirs = 4

        self._Initialize(boardPos)

    def GetVisited(self):
        return self.visited

    def SetVisited(self, visited):
        self.visited = visited

    def GetNeighbor(self, index):
        return self.neighbors[index]

    def GetVertices(self):
        return self.vertices

    def GetIndices(self):
        return self.indices

    def GetWallVertices(self):
        return self.wallVertices

    def GetWallIndices(self):
        return self.wallIndices

    def GetPos(self):
        return self.pos

    def GetNonVisitedNeighbor(self):
        nonVisitedNeighbors = []

        for i in range(self.numDirs):
            if self.neighbors[i] != None and self.neighbors[i].GetVisited() == False:
                nonVisitedNeighbors.append(self.neighbors[i])

        if len(nonVisitedNeighbors) > 0:
            randIndex = random.randrange(0, len(nonVisitedNeighbors))

            return nonVisitedNeighbors[randIndex]

        return None

    def SetNeighbors(self, allCells, gridSize):
        c = int(self.pos[0])
        r = int(self.pos[1])

        cellOffset = (gridSize[1] - 1 - r) * gridSize[0] + c

        if r - 1 >= 0:
            self.neighbors.append(allCells[cellOffset + gridSize[0]])
        else:
            self.neighbors.append(None)

        if c + 1 < gridSize[0]:
            self.neighbors.append(allCells[cellOffset + 1])
        else:
            self.neighbors.append(None)

        if r + 1 < gridSize[1]:
            self.neighbors.append(allCells[cellOffset - gridSize[0]])
        else:
            self.neighbors.append(None)

        if c - 1 >= 0:
            self.neighbors.append(allCells[cellOffset - 1])
        else:
            self.neighbors.append(None)

    def IsWall(self, index):
        return self.walls[index]

    def RemoveWall(self, index):
        self.walls[index] = False

        if index == 0:
            self.wallIndices[index * 2 + 1] = index
        elif index == 1:
            self.wallIndices[index * 2 + 1] = index
        elif index == 2:
            self.wallIndices[index * 2 + 1] = index
        elif index == 3:
            self.wallIndices[index * 2 + 1] = index
    
    def _Initialize(self, boardPos):
        posX = boardPos[0][0] + self.pos[0] * self.size[0]
        posY = boardPos[0][1] + self.pos[1] * self.size[1]

        verticesData = [
            posX, posY, 0.0, 0.0, 0.0, 0.0, 1.0,
            posX + self.size[0], posY, 0.0, 0.0, 0.0, 0.0, 1.0,
            posX + self.size[0], posY + self.size[1], 0.0, 0.0, 0.0, 0.0, 1.0,
            posX, posY + self.size[1], 0.0, 0.0, 0.0, 0.0, 1.0
            ]

        indicesData = [
            0, 1, 2, 2, 3, 0
            ]

        self.vertices = np.array(verticesData, dtype = np.float32)
        self.indices = np.array(indicesData, dtype = np.uint32)

        wallVerticesData = [
            posX, posY, 1.0, 1.0, 1.0, 1.0, 0.5,
            posX + self.size[0], posY, 1.0, 1.0, 1.0, 1.0, 0.5,
            posX + self.size[0], posY + self.size[1], 1.0, 1.0, 1.0, 1.0, 0.5,
            posX, posY + self.size[1], 1.0, 1.0, 1.0, 1.0, 0.5
            ]

        wallIndicesData = []

        if self.walls[0] == True:
            wallIndicesData.append(0)
            wallIndicesData.append(1)
        else:
            wallIndicesData.append(0)
            wallIndicesData.append(0)

        if self.walls[1] == True:
            wallIndicesData.append(1)
            wallIndicesData.append(2)
        else:
            wallIndicesData.append(1)
            wallIndicesData.append(1)

        if self.walls[2] == True:
            wallIndicesData.append(2)
            wallIndicesData.append(3)
        else:
            wallIndicesData.append(2)
            wallIndicesData.append(2)

        if self.walls[3] == True:
            wallIndicesData.append(3)
            wallIndicesData.append(0)
        else:
            wallIndicesData.append(3)
            wallIndicesData.append(3)

        self.wallVertices = np.array(wallVerticesData, dtype = np.float32)
        self.wallIndices = np.array(wallIndicesData, dtype = np.uint32)    

class MazeGenerator:
    def __init__(self, displaySize, programName):
        self.programName = programName

        self.backgroundStuffVerticesList = []
        self.backgroundStuffIndicesList = []

        self.backgroundVertices = []
        self.backgroundIndices = []

        self.backgroundLineVertices = []
        self.backgroundLineIndices = []

        self.gameStuffVerticesList = []
        self.gameStuffIndicesList = []

        self.gridVertices = []
        self.gridIndices = []

        self.allCellsVertices = []
        self.allCellsIndices = []

        self.curCellVertices = []
        self.curCellIndices = []

        self.backgroundModelMat = glm.mat4()
        self.gameModelMat = glm.mat4()

        self.dirty = [True, True]

        self.boardSize = (560, 560)
        self.boardPos = []

        self.displaySize = (displaySize[0], displaySize[1])

        self.allCells = []
        self.gridSize = []

        self.cellSize = [40, 40]

        self.startCell = None
        self.goalCell = None

        self.curCell = None

        self.visitedCells = []

        self.pathfindingDir = {'FRONT' : 2, 'LEFT' : 3, 'BACK' : 0, 'RIGHT' : 1}
        self.traceDir = []

        self.generationStates = {'READY' : True, 'PROCESS' : False, 'COMPLETION' : False}
        self.pathfindingStates = {'READY' : True, 'PROCESS' : False, 'COMPLETION' : False}

        self.elapsedTimeBeforePathfinding = 0.0
        self.waitingTimeBeforePathfinding = 1.0

        self.numVertexComponents = 7
        self.numCellVertices = 4
        self.numBackgroundStuff = 2
        self.numGameStuff = 3

        self.backgroundVAO = glGenVertexArrays(self.numBackgroundStuff)
        self.backgroundVBO = glGenBuffers(self.numBackgroundStuff)
        self.backgroundEBO = glGenBuffers(self.numBackgroundStuff)

        self.gameVAO = glGenVertexArrays(self.numGameStuff)
        self.gameVBO = glGenBuffers(self.numGameStuff)
        self.gameEBO = glGenBuffers(self.numGameStuff)

        self._InitializeBackgroundStuff()

        self._InitializeGameStuff()

    def Restart(self):
        self._InitializeGameStuff()

    def UpdateAboutKeyInput(self, key, value = True):
        if key == glfw.KEY_3:
            self.cellSize[0] *= 2
            self.cellSize[1] *= 2

            if self.cellSize[0] > 80:
                self.cellSize[0] = 80
                self.cellSize[1] = 80

            self.Restart()

        elif key == glfw.KEY_4:
            self.cellSize[0] /= 2
            self.cellSize[1] /= 2

            if self.cellSize[0] < 5:
                self.cellSize[0] = 5
                self.cellSize[1] = 5

            self.Restart()

    def UpdateAboutMouseInput(self, button, pos):
        if gSceneManager.GetScreenControlWithMouse() == False:
            return

    def Update(self, deltaTime):
        if self.curCell == None:
            if self.generationStates['COMPLETION'] == False:                
                self._UpdateWhenGenerationCompletion(deltaTime)
            elif self.pathfindingStates['COMPLETION'] == False:                
                self._UpdateWhenPathfindingCompletion()
            
            return

        if self.generationStates['PROCESS'] == True:
            self._GenerateMaze()

        if self.pathfindingStates['PROCESS'] == True:
            self._PathfindingMaze()

    def Draw(self):
        self._DrawBackground()

        self._DrawAllCells()

        self._DrawGrid()

        if self.generationStates['PROCESS'] == True or self.pathfindingStates['PROCESS'] == True:
            self._DrawCurCell()       

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
            4, 5, 6, 6, 7, 4,
            8, 9, 10, 10, 11, 8,
            12, 13, 14, 14, 15, 12
            ]

        self.backgroundVertices = np.array(backgroundVerticesData, dtype = np.float32)
        self.backgroundIndices = np.array(backgroundIndicesData, dtype = np.uint32)

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

        self.backgroundStuffVerticesList.clear()
        self.backgroundStuffIndicesList.clear()

        self.backgroundStuffVerticesList.append(self.backgroundVertices)
        self.backgroundStuffVerticesList.append(self.backgroundLineVertices)

        self.backgroundStuffIndicesList.append(self.backgroundIndices)
        self.backgroundStuffIndicesList.append(self.backgroundLineIndices)

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
        gSceneManager.ClearSpecificProgramArgs()
        
        for key in self.generationStates.keys():
            self.generationStates[key] = False
        self.generationStates['READY'] = True

        for key in self.pathfindingStates.keys():
            self.pathfindingStates[key] = False
        self.pathfindingStates['READY'] = True

        self.elapsedTimeBeforePathfinding = 0.0

        self.gridSize.clear()
        self.allCells.clear()
        self.visitedCells.clear()

        self.gridSize.append(int(self.boardSize[0] / self.cellSize[0]))
        self.gridSize.append(int(self.boardSize[1] / self.cellSize[1]))

        for r in range(self.gridSize[1] - 1, -1, -1):
            for c in range(self.gridSize[0]):
                self.allCells.append(Cell((c, r), (self.cellSize[0], self.cellSize[1]), self.boardPos))

        for i in range(len(self.allCells)):
            self.allCells[i].SetNeighbors(self.allCells, self.gridSize)

        self.curCell = self.allCells[0]

        cellVerticesInterval = self.numVertexComponents * self.numCellVertices
        cellWallIndicesInterval = self.numCellVertices * 2

        gridVerticesSize = len(self.allCells) * cellVerticesInterval
        gridIndicesSize = len(self.allCells) * cellWallIndicesInterval

        self.gridVertices = np.zeros(gridVerticesSize, dtype = np.float32)
        self.gridIndices = np.zeros(gridIndicesSize, dtype = np.uint32)

        for i in range(len(self.allCells)):
            vOffset = i * cellVerticesInterval
            iOffset = i * cellWallIndicesInterval
            viOffset = i * self.numCellVertices

            self.gridVertices[vOffset : vOffset + cellVerticesInterval] = self.allCells[i].GetWallVertices()
            self.gridIndices[iOffset : iOffset + cellWallIndicesInterval] = self.allCells[i].GetWallIndices() + viOffset

        cellIndicesInterval = 6

        allCellsVerticesSize = len(self.allCells) * cellVerticesInterval
        allCellsIndicesSize = len(self.allCells) * cellIndicesInterval

        self.allCellsVertices = np.zeros(allCellsVerticesSize, dtype = np.float32)
        self.allCellsIndices = np.zeros(allCellsIndicesSize, dtype = np.uint32)

        for i in range(len(self.allCells)):
            vOffset = i * cellVerticesInterval
            iOffset = i * cellIndicesInterval
            viOffset = i * self.numCellVertices

            self.allCellsVertices[vOffset : vOffset + cellVerticesInterval] = self.allCells[i].GetVertices()
            self.allCellsIndices[iOffset : iOffset + cellIndicesInterval] = self.allCells[i].GetIndices() + viOffset

        curCellVerticesData = []
        curCellIndicesData = []

        for i in range(self.numCellVertices):
            curCellPosX = self.curCell.GetVertices()[self.numVertexComponents * i + 0]
            curCellPosY = self.curCell.GetVertices()[self.numVertexComponents * i + 1]

            curCellVerticesData.append(curCellPosX)
            curCellVerticesData.append(curCellPosY)
            curCellVerticesData.append(2.0)

            curCellVerticesData.append(0.0)
            curCellVerticesData.append(1.0)
            curCellVerticesData.append(0.0)
            curCellVerticesData.append(1.0)

        for i in range(len(self.curCell.GetIndices())):
            curCellIndicesData.append(self.curCell.GetIndices()[i])

        self.curCellVertices = np.array(curCellVerticesData, dtype = np.float32)
        self.curCellIndices = np.array(curCellIndicesData, dtype = np.uint32)

        self.gameStuffVerticesList.clear()
        self.gameStuffIndicesList.clear()

        self.gameStuffVerticesList.append(self.gridVertices)
        self.gameStuffVerticesList.append(self.allCellsVertices)
        self.gameStuffVerticesList.append(self.curCellVertices)

        self.gameStuffIndicesList.append(self.gridIndices)
        self.gameStuffIndicesList.append(self.allCellsIndices)
        self.gameStuffIndicesList.append(self.curCellIndices)

        for i in range(self.numGameStuff):
            glBindVertexArray(self.gameVAO[i])

            glBindBuffer(GL_ARRAY_BUFFER, self.gameVBO[i])

            if i == 0:
                glBufferData(GL_ARRAY_BUFFER, self.gameStuffVerticesList[i].nbytes, self.gameStuffVerticesList[i], GL_STATIC_DRAW)
            else:
                glBufferData(GL_ARRAY_BUFFER, self.gameStuffVerticesList[i].nbytes, self.gameStuffVerticesList[i], GL_DYNAMIC_DRAW)

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.gameEBO[i])

            if i == 0:
                glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.gameStuffIndicesList[i].nbytes, self.gameStuffIndicesList[i], GL_DYNAMIC_DRAW)
            else:
                glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.gameStuffIndicesList[i].nbytes, self.gameStuffIndicesList[i], GL_STATIC_DRAW)

            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.gameStuffVerticesList[i].itemsize * self.numVertexComponents, ctypes.c_void_p(0))

            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, self.gameStuffVerticesList[i].itemsize * self.numVertexComponents, ctypes.c_void_p(self.gameStuffVerticesList[i].itemsize * 3))

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        gridSizeStr = ".Grid Size : " + str(self.gridSize[0]) + " X " + str(self.gridSize[1])

        generationStateStr = ".  Generation : Ready"
        pathfindingStateStr = ".  Pathfinding : Ready"

        gSceneManager.AddSpecificProgramArgs(gSceneManager.GetColor('DefaultColor_', 7), [585, 570], 0, 1, 'Large', gridSizeStr)

        gSceneManager.AddSpecificProgramArgs(gSceneManager.GetColor('DefaultColor_', 7), [585, 535], 20, 2, 'Medium', generationStateStr, pathfindingStateStr)
        
        gSceneManager.AddSpecificProgramArgs(gSceneManager.GetColor('DefaultColor_', 1), [670, 16], 0, 1, 'Medium', self.programName)

        for key in self.generationStates.keys():
            self.generationStates[key] = False
        self.generationStates['PROCESS'] = True

        generationStateStr = ".  Generation : Process"
        gSceneManager.SetSpecificProgramArgs(1, Index.SPECIFIC_PROGRAM_INFO_TEXT_0 + 0, generationStateStr)

    def _UpdateWhenGenerationCompletion(self, deltaTime):
        self.elapsedTimeBeforePathfinding += deltaTime

        if self.elapsedTimeBeforePathfinding < self.waitingTimeBeforePathfinding:
            return

        self.startCell = self.allCells[0]
        self.goalCell = self.allCells[len(self.allCells) - 1]

        self.curCell = self.startCell

        self._UpdateCell(self.startCell, [1.0, 1.0, 0.0, 1.0], False)
        self._UpdateCell(self.goalCell, [0.0, 0.0, 1.0, 1.0], False)

        for i in range(self.numCellVertices):
            self.curCellVertices[self.numVertexComponents * i + 3] = 1.0
            self.curCellVertices[self.numVertexComponents * i + 4] = 0.0
            self.curCellVertices[self.numVertexComponents * i + 5] = 0.0
            self.curCellVertices[self.numVertexComponents * i + 6] = 1.0

        glBindBuffer(GL_ARRAY_BUFFER, self.gameVBO[2])
        glBufferSubData(GL_ARRAY_BUFFER, 0, self.gameStuffVerticesList[2].nbytes, self.gameStuffVerticesList[2])

        glBindBuffer(GL_ARRAY_BUFFER, 0)

        for key in self.generationStates.keys():
            self.generationStates[key] = False
        self.generationStates['COMPLETION'] = True

        for key in self.pathfindingStates.keys():
            self.pathfindingStates[key] = False
        self.pathfindingStates['PROCESS'] = True

        generationStateStr = ".  Generation : Complete"
        gSceneManager.SetSpecificProgramArgs(1, Index.SPECIFIC_PROGRAM_INFO_TEXT_0 + 0, generationStateStr)

        pathfindingStateStr = ".  Pathfinding : Process"
        gSceneManager.SetSpecificProgramArgs(1, Index.SPECIFIC_PROGRAM_INFO_TEXT_0 + 1, pathfindingStateStr)

    def _UpdateWhenPathfindingCompletion(self):
        for key in self.pathfindingStates.keys():
            self.pathfindingStates[key] = False
        self.pathfindingStates['COMPLETION'] = True

        pathfindingStateStr = ".  Pathfinding : Complete"
        gSceneManager.SetSpecificProgramArgs(1, Index.SPECIFIC_PROGRAM_INFO_TEXT_0 + 1, pathfindingStateStr)

    def _UpdateCell(self, cell, cellColor, curCellUpdate = False):
        if cell != None:
            cellPos = cell.GetPos()

            cellVerticesInterval = self.numVertexComponents * self.numCellVertices
            rOffset = int((self.gridSize[1] - 1 - cellPos[1]) * self.gridSize[0] * cellVerticesInterval)
            cOffset = int(cellPos[0] * cellVerticesInterval)

            if cellColor == None:
                cellColor = [0.5, 0.0, 0.0, 0.5]

            for v in range(self.numCellVertices):
                vOffset = v * self.numVertexComponents

                self.allCellsVertices[rOffset + cOffset + vOffset + 3] = cellColor[0]
                self.allCellsVertices[rOffset + cOffset + vOffset + 4] = cellColor[1]
                self.allCellsVertices[rOffset + cOffset + vOffset + 5] = cellColor[2]
                self.allCellsVertices[rOffset + cOffset + vOffset + 6] = cellColor[3]

            self.dirty[1] = True

        if curCellUpdate == True:
            curCellVerticesData = self.curCell.GetVertices()

            for i in range(self.numCellVertices):
                curCellPosX = curCellVerticesData[self.numVertexComponents * i + 0]
                curCellPosY = curCellVerticesData[self.numVertexComponents * i + 1]

                self.curCellVertices[self.numVertexComponents * i + 0] = curCellPosX
                self.curCellVertices[self.numVertexComponents * i + 1] = curCellPosY

            glBindBuffer(GL_ARRAY_BUFFER, self.gameVBO[2])
            glBufferSubData(GL_ARRAY_BUFFER, 0, self.gameStuffVerticesList[2].nbytes, self.gameStuffVerticesList[2])

            glBindBuffer(GL_ARRAY_BUFFER, 0)

    def _GenerateMaze(self):
        if self.curCell.GetVisited() == False:
            self.curCell.SetVisited(True)

            self._UpdateCell(self.curCell, None, True)

        else:
            self._UpdateCell(None, None, True)

        nextCell = self.curCell.GetNonVisitedNeighbor()

        if nextCell != None:
            self.visitedCells.append(self.curCell)

            self._RemoveWalls(nextCell)

            self.curCell = nextCell

        elif len(self.visitedCells) > 0:
            self.curCell = self.visitedCells.pop()

        else:
            self.curCell = None

    def _RemoveWalls(self, nextCell):
        curCellPos = self.curCell.GetPos()
        nextCellPos = nextCell.GetPos()

        if nextCellPos[0] - curCellPos[0] > 0:
            self.curCell.RemoveWall(1)
            nextCell.RemoveWall(3)
        elif nextCellPos[0] - curCellPos[0] < 0:
            self.curCell.RemoveWall(3)
            nextCell.RemoveWall(1)

        if nextCellPos[1] - curCellPos[1] > 0:
            self.curCell.RemoveWall(2)
            nextCell.RemoveWall(0)
        elif nextCellPos[1] - curCellPos[1] < 0:
            self.curCell.RemoveWall(0)
            nextCell.RemoveWall(2)

        cellWallIndicesInterval = self.numCellVertices * 2
        rOffset = int((self.gridSize[1] - 1 - curCellPos[1]) * self.gridSize[0] * cellWallIndicesInterval)
        cOffset = int(curCellPos[0] * cellWallIndicesInterval)

        for i in range(len(self.curCell.GetWallIndices())):
            self.gridIndices[rOffset + cOffset + i] = self.gridIndices[rOffset + cOffset + 0] + self.curCell.GetWallIndices()[i]

        rOffset = int((self.gridSize[1] - 1 - nextCellPos[1]) * self.gridSize[0] * cellWallIndicesInterval)
        cOffset = int(nextCellPos[0] * cellWallIndicesInterval)

        for i in range(len(nextCell.GetWallIndices())):
            self.gridIndices[rOffset + cOffset + i] = self.gridIndices[rOffset + cOffset + 0] + nextCell.GetWallIndices()[i]

        self.dirty[0] = True

    def _PathfindingMaze(self):
        leftDir = self.pathfindingDir['LEFT']
        dirDiff = 0

        if self.curCell.IsWall(leftDir) == True:
            frontDir = self.pathfindingDir['FRONT']

            if self.curCell.IsWall(frontDir) == True:
                keyList = list(self.pathfindingDir.keys())
                valueList = list(self.pathfindingDir.values())
                valueListRotatedCW = [valueList[-1]] + valueList[ : -1]
                self.pathfindingDir = dict(zip(keyList, valueListRotatedCW))
            else:
                if len(self.traceDir) > 0:
                    dirDiff = frontDir - self.traceDir[-1]

                if dirDiff != 0 and abs(dirDiff) % 2 == 0:
                    self.traceDir.pop()
                    self._UpdateCell(self.curCell, [1.0, 1.0, 1.0, 0.5], True)
                else:
                    self.traceDir.append(frontDir)

                    if self.curCell == self.startCell:
                        self._UpdateCell(None, None, True)
                    else:
                        self._UpdateCell(self.curCell, [0.0, 1.0, 0.0, 0.5], True)

                self.curCell = self.curCell.GetNeighbor(frontDir)

        else:
            keyList = list(self.pathfindingDir.keys())
            valueList = list(self.pathfindingDir.values())
            valueListRotatedCCW = valueList[1 : ] + [valueList[0]]
            self.pathfindingDir = dict(zip(keyList, valueListRotatedCCW))

            frontDir = self.pathfindingDir['FRONT']

            if len(self.traceDir) > 0:
                dirDiff = frontDir - self.traceDir[-1]

            if dirDiff != 0 and abs(dirDiff) % 2 == 0:
                self.traceDir.pop()
                self._UpdateCell(self.curCell, [1.0, 1.0, 1.0, 0.5], True)
            else:
                self.traceDir.append(frontDir)

                if self.curCell == self.startCell:
                    self._UpdateCell(None, None, True)
                else:
                    self._UpdateCell(self.curCell, [0.0, 1.0, 0.0, 0.5], True)

            self.curCell = self.curCell.GetNeighbor(frontDir)

        if self.goalCell == self.curCell:          
            self.curCell = None

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

        glLineWidth(2.0)

        glBindVertexArray(self.backgroundVAO[backgroundStuffIndex])
        glDrawElements(GL_LINES, len(self.backgroundStuffIndicesList[backgroundStuffIndex]), GL_UNSIGNED_INT, None)

        glBindVertexArray(0)

        glPopAttrib()

    def _DrawAllCells(self):
        glPushAttrib(GL_COLOR_BUFFER_BIT | GL_ENABLE_BIT)

        glEnable(GL_BLEND)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        shader = gSceneManager.GetShader()
        shader.SetMat4('modelMat', self.gameModelMat)

        if self.dirty[1] == True:
            glBindBuffer(GL_ARRAY_BUFFER, self.gameVBO[1])
            glBufferSubData(GL_ARRAY_BUFFER, 0, self.gameStuffVerticesList[1].nbytes, self.gameStuffVerticesList[1])

            glBindBuffer(GL_ARRAY_BUFFER, 0)

            self.dirty[1] = False

        glBindVertexArray(self.gameVAO[1])
        glDrawElements(GL_TRIANGLES, len(self.gameStuffIndicesList[1]), GL_UNSIGNED_INT, None)

        glBindVertexArray(0)

        glPopAttrib()

    def _DrawGrid(self):
        glPushAttrib(GL_COLOR_BUFFER_BIT | GL_ENABLE_BIT | GL_LINE_BIT)

        glEnable(GL_BLEND)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        shader = gSceneManager.GetShader()
        shader.SetMat4('modelMat', self.gameModelMat)

        if self.dirty[0] == True:
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.gameEBO[0])
            glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, self.gameStuffIndicesList[0].nbytes, self.gameStuffIndicesList[0])

            self.dirty[0] = False

        glLineWidth(2.0)

        glBindVertexArray(self.gameVAO[0])
        glDrawElements(GL_LINES, len(self.gameStuffIndicesList[0]), GL_UNSIGNED_INT, None)

        glBindVertexArray(0)

        glPopAttrib()

    def _DrawCurCell(self):
        glPushAttrib(GL_COLOR_BUFFER_BIT | GL_ENABLE_BIT)

        glEnable(GL_BLEND)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        shader = gSceneManager.GetShader()
        shader.SetMat4('modelMat', self.gameModelMat)

        glBindVertexArray(self.gameVAO[2])
        glDrawElements(GL_TRIANGLES, len(self.gameStuffIndicesList[2]), GL_UNSIGNED_INT, None)

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
        elif key == glfw.KEY_SPACE:
            gInputManager.SetKeyState(glfw.KEY_SPACE, True)

        if key == glfw.KEY_1:
            gInputManager.SetKeyState(glfw.KEY_1, True)
        elif key == glfw.KEY_2:
            gInputManager.SetKeyState(glfw.KEY_2, True)
        elif key == glfw.KEY_3:
            gInputManager.SetKeyState(glfw.KEY_3, True)
        elif key == glfw.KEY_4:
            gInputManager.SetKeyState(glfw.KEY_4, True)
        elif key == glfw.KEY_8:
            gInputManager.SetKeyState(glfw.KEY_8, True)
        elif key == glfw.KEY_9:
            gInputManager.SetKeyState(glfw.KEY_9, True)
        elif key == glfw.KEY_0:
            gInputManager.SetKeyState(glfw.KEY_0, True)

        if key == glfw.KEY_B:
            gInputManager.SetKeyState(glfw.KEY_B, True)
        elif key == glfw.KEY_C:
            gInputManager.SetKeyState(glfw.KEY_C, True)
        elif key == glfw.KEY_E:
            gInputManager.SetKeyState(glfw.KEY_E, True)
        elif key == glfw.KEY_F:
            gInputManager.SetKeyState(glfw.KEY_F, True)
        elif key == glfw.KEY_G:
            gInputManager.SetKeyState(glfw.KEY_G, True)
        elif key == glfw.KEY_I:
            gInputManager.SetKeyState(glfw.KEY_I, True) 
        elif key == glfw.KEY_P:
            gInputManager.SetKeyState(glfw.KEY_P, True)            
        elif key == glfw.KEY_R:
            gInputManager.SetKeyState(glfw.KEY_R, True)
        elif key == glfw.KEY_V:
            gInputManager.SetKeyState(glfw.KEY_V, True)

        if key == glfw.KEY_W:
            gInputManager.SetKeyState(glfw.KEY_W, True)
        elif key == glfw.KEY_S:
            gInputManager.SetKeyState(glfw.KEY_S, True)
        elif key == glfw.KEY_A:
            gInputManager.SetKeyState(glfw.KEY_A, True)
        elif key == glfw.KEY_D:
            gInputManager.SetKeyState(glfw.KEY_D, True)        

        if key == glfw.KEY_LEFT:
            gInputManager.SetKeyState(glfw.KEY_LEFT, True)
        elif key == glfw.KEY_RIGHT:
            gInputManager.SetKeyState(glfw.KEY_RIGHT, True)
        elif key == glfw.KEY_UP:
            gInputManager.SetKeyState(glfw.KEY_UP, True)
        elif key == glfw.KEY_DOWN:
            gInputManager.SetKeyState(glfw.KEY_DOWN, True)

    if action == glfw.RELEASE:
        if key == glfw.KEY_W:
            gInputManager.SetKeyState(glfw.KEY_W, False)
        elif key == glfw.KEY_S:
            gInputManager.SetKeyState(glfw.KEY_S, False)
        elif key == glfw.KEY_A:
            gInputManager.SetKeyState(glfw.KEY_A, False)
        elif key == glfw.KEY_D:
            gInputManager.SetKeyState(glfw.KEY_D, False)

        if key == glfw.KEY_LEFT:
            gInputManager.SetKeyState(glfw.KEY_LEFT, False)
        elif key == glfw.KEY_RIGHT:
            gInputManager.SetKeyState(glfw.KEY_RIGHT, False)
        elif key == glfw.KEY_UP:
            gInputManager.SetKeyState(glfw.KEY_UP, False)
        elif key == glfw.KEY_DOWN:
            gInputManager.SetKeyState(glfw.KEY_DOWN, False)

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
    projectName = "Maze Generator.Part 1"
    programName = "# MazeGenerator"

    if not glfw.init():
        return

    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)

    glfwWindow = glfw.create_window(displaySize[0], displaySize[1], projectName, None, None)

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
    gSceneManager.AddObject(MazeGenerator(displaySize, programName))
    
    lastElapsedTime = glfw.get_time()
    deltaTime = 0.0

    while glfw.window_should_close(glfwWindow) == False:
        glfw.poll_events()        

        gSceneManager.Update(deltaTime)        

        if gSceneManager.Draw(deltaTime) == True:
            glfw.swap_buffers(glfwWindow)

        gSceneManager.PostUpdate(deltaTime)        

        deltaTime = glfw.get_time() - lastElapsedTime
        lastElapsedTime = glfw.get_time()


glfw.terminate()


if __name__ == "__main__":
    Main()    