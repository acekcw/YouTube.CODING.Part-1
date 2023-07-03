
from OpenGL.GL import *
from OpenGL.GL.shaders import *

import glfw
import glm
import math

import numpy as np
import freetype as ft


vertexShaderCode = """

# version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec4 aColor;
layout(location = 3) in mat4 aInstanceMat;

out vec4 color;

uniform mat4 prjMat;
uniform mat4 viewMat;
uniform mat4 modelMat;

uniform bool useInstance;

void main()
{
    if (useInstance)
    {
        gl_Position = prjMat * viewMat * modelMat * aInstanceMat * vec4(aPos, 1.0);
    }
    else
    {
        gl_Position = prjMat * viewMat * modelMat * vec4(aPos, 1.0);
    }

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
        self.displaySize = (800, 600)
        self.screenSize = (560, 560)        
        self.screenPos = []        

        self.programInfoAreaVertices = []
        self.programInfoAreaIndices = []

        self.fovy = 45.0
        self.aspect = self.screenSize[0] / self.screenSize[1]
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
        self.view2DMat = glm.inverse(self.view2DMat)        

        self.programInfoAreaModelMat = glm.mat4() 

        self.screenControlWithMouse = False
        self.ableMouseDragged = False
        self.ableMouseEscape = False

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

    def GetView3D(self):
        return self.view3D

    def SetView3D(self, view3D):
        self.view3D = view3D

    def GetPerspectivePrjMat(self):
        return self.perspectivePrjMat

    def GetOrthoPrjMat(self):
        return self.orthoPrjMat

    def GetView3DMat(self):
        self.view3DMat = self.camera.GetViewMat()
        return self.view3DMat

    def GetView2DMat(self):
        return self.view2DMat

    def GetPause(self):
        return self.pause    

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

    def GetScreenPos(self):
        return self.screenPos

    def GetScreenSize(self):
        return self.screenSize

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
        if gInputManager.GetKeyState(glfw.KEY_Q) == True:
            self.camera.ProcessKeyboard('UPWARD', 0.05)
            self.dirty = True
        if gInputManager.GetKeyState(glfw.KEY_E) == True:
            self.camera.ProcessKeyboard('DOWNWARD', 0.05)
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

        self.screenPos.clear()

        screenLbPos = [20, 20]
        screenRtPos = []
        screenRtPos.append(screenLbPos[0] + self.screenSize[0])
        screenRtPos.append(screenLbPos[1] + self.screenSize[1])

        self.screenPos.append(screenLbPos)
        self.screenPos.append(screenRtPos)

        self.debugMat = glGetFloatv(GL_MODELVIEW_MATRIX)

    def MakeFont(self, fontPath = None):
        if fontPath == None:
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
            gInputManager.SetMouseEntered(False)
            gInputManager.SetKeyState(glfw.KEY_C, False)        
        if gInputManager.GetKeyState(glfw.KEY_F) == True:
            self.specificProgramInfo = not self.specificProgramInfo
            gInputManager.SetKeyState(glfw.KEY_F, False)
        if gInputManager.GetKeyState(glfw.KEY_G) == True:
            self.ableMouseDragged = not self.ableMouseDragged
            gInputManager.SetKeyState(glfw.KEY_G, False)
        if gInputManager.GetKeyState(glfw.KEY_H) == True:
            self.ableMouseEscape = not self.ableMouseEscape
            gInputManager.SetKeyState(glfw.KEY_H, False)
        if gInputManager.GetKeyState(glfw.KEY_I) == True:
            self.programInfo = not self.programInfo                
            gInputManager.SetKeyState(glfw.KEY_I, False)
        if gInputManager.GetKeyState(glfw.KEY_P) == True:
            self.pause = not self.pause
            gInputManager.SetMouseEntered(False)
            gInputManager.SetKeyState(glfw.KEY_P, False)
        if gInputManager.GetKeyState(glfw.KEY_R) == True:
            for i in range(numObjects):
                self.objects[i].Restart()
            gInputManager.SetKeyState(glfw.KEY_R, False)
        if gInputManager.GetKeyState(glfw.KEY_V) == True:
            self.view3D = not self.view3D
            for i in range(numObjects):
                self.objects[i].Restart()
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

        if self.view3D == True:
            self.SetCameraPos()
            self.SailCamera()

        if self.pause == True:
            return

        numObjects = len(self.objects)

        for i in range(numObjects):
            if self.controlFPS == True:
                self.objects[i].Update(self.elapsedTime)
            else:
                self.objects[i].Update(deltaTime)        

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

        if self.view3D == True:
            self.screenControlWithMouse = True
            self.ableMouseDragged = False
            self.ableMouseEscape = False
       
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

        infoText.append('.  MouseEscape(H) : ')
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
    def __init__(self, cameraPos = None):
        if cameraPos == None:
            self.cameraPos = glm.vec3(0.0, 0.0, 10.0)
        else:
            self.cameraPos = cameraPos
            
        self.cameraFront = glm.vec3(0.0, 0.0, -1.0)
        self.cameraUp = glm.vec3(0.0, 1.0, 0.0)
        self.cameraRight = glm.vec3(1.0, 0.0, 0.0)
        self.cameraWorldUp = glm.vec3(0.0, 1.0, 0.0)

        self.pitch = 0.0
        self.yaw = 180.0

        self.mouseSensitivity = 0.1

        self.UpdateCameraVectors()

    def GetPos(self):
        return self.cameraPos

    def SetPos(self, cameraPos):
        self.cameraPos = cameraPos

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
        elif direction == "UPWARD":
            self.cameraPos += self.cameraUp * velocity
        elif direction == "DOWNWARD":
            self.cameraPos -= self.cameraUp * velocity

    def UpdateCameraVectors(self):
        self.cameraFront.x = math.sin(glm.radians(self.yaw)) * math.cos(glm.radians(self.pitch))
        self.cameraFront.y = math.sin(glm.radians(self.pitch))
        self.cameraFront.z = math.cos(glm.radians(self.yaw)) * math.cos(glm.radians(self.pitch))

        self.cameraFront = glm.normalize(self.cameraFront)

        self.cameraRight = glm.normalize(glm.cross(self.cameraWorldUp, self.cameraFront))
        self.cameraUp = glm.normalize(glm.cross(self.cameraFront, self.cameraRight))

gSceneManager = SceneManager(True)
gInputManager = InputManager()

class TestProgram:
    def __init__(self, programName):
        self.programName = programName       

        self.drawingStuffVerticesList = []        
        self.drawingStuffIndicesList = []

        self.object1sVertices = []
        self.objectsIndices = []

        self.GUIStuffVerticesList = []
        self.GUIStuffIndicesList = []

        self.backgroundVertices = []
        self.backgroundIndices = []
        
        self.backgroundLineVertices = []
        self.backgroundLineIndices = []

        self.rotDegree = 0.0
        
        self.drawingModelMat = glm.mat4()
        self.GUIModelMat = glm.mat4()
        
        self.numVertexComponents = 7
        self.numVertexComponentsWithTexCoord = 9        
        self.numDrawingStuffIn3D = 2
        self.numDrawingStuffIn2D = 2
        self.numGUIStuff = 2

        self.drawingStuffVAO = None
        self.drawingStuffVBO = None
        self.drawingStuffEBO = None

        self.GUIVAO = glGenVertexArrays(self.numGUIStuff)
        self.GUIVBO = glGenBuffers(self.numGUIStuff)
        self.GUIEBO = glGenBuffers(self.numGUIStuff)

        self._Initialize()        

    def Restart(self):
        if len(self.drawingStuffVAO) > 0:
            glDeleteVertexArrays(len(self.drawingStuffVAO), self.drawingStuffVAO)        
        if len(self.drawingStuffVBO) > 0:
            glDeleteBuffers(len(self.drawingStuffVBO), self.drawingStuffVBO)            
        if len(self.drawingStuffEBO) > 0:
            glDeleteBuffers(len(self.drawingStuffEBO), self.drawingStuffEBO)

        self._Initialize()        

    def UpdateAboutKeyInput(self, key, value = True):
        pass

    def UpdateAboutMouseInput(self, button, pos):
        pass            

    def Update(self, deltaTime):
        if gSceneManager.GetView3D() == True:
            self._Update3D(deltaTime)

    def Draw(self):
        shader = gSceneManager.GetShader()

        displaySize = gSceneManager.GetDisplaySize()
        screenPos = gSceneManager.GetScreenPos()
        screenSize = gSceneManager.GetScreenSize()

        if gSceneManager.GetView3D() == True:
            shader.SetMat4('prjMat', gSceneManager.GetPerspectivePrjMat())
            shader.SetMat4('viewMat', gSceneManager.GetView3DMat())

            glViewport(screenPos[0][0], screenPos[0][1], screenSize[0], screenSize[1])

            self._DrawDrawingStuffIn3D()

        else:
            shader.SetMat4('prjMat', gSceneManager.GetOrthoPrjMat())
            shader.SetMat4('viewMat', gSceneManager.GetView2DMat())            

            self._DrawDrawingStuffIn2D()
            
        glViewport(0, 0, displaySize[0], displaySize[1])
        
        shader.SetMat4('prjMat', gSceneManager.GetOrthoPrjMat())
        shader.SetMat4('viewMat', gSceneManager.GetView2DMat())

        self._DrawGUI()

    def _Initialize(self):
        self.drawingModelMat = glm.mat4()
        self.GUIModelMat = glm.mat4()
        
        if gSceneManager.GetView3D() == True:
            self._InitializeDrawingStuffIn3D()
        else:
            self._InitializeDrawingStuffIn2D()

        self._InitializeGUIStuff()

    def _InitializeDrawingStuffIn3D(self):
        self.drawingStuffVAO = glGenVertexArrays(self.numDrawingStuffIn3D)
        self.drawingStuffVBO = glGenBuffers(self.numDrawingStuffIn3D)
        self.drawingStuffEBO = glGenBuffers(self.numDrawingStuffIn3D)        

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

        self.object1sVertices = np.array(cubeVerticesData, dtype = np.float32)
        self.object1sIndices = np.array(cubeIndicesData, dtype = np.uint32)

        self.drawingStuffVerticesList.clear()
        self.drawingStuffIndicesList.clear()

        self.drawingStuffVerticesList.append(self.object1sVertices)
        
        self.drawingStuffIndicesList.append(self.object1sIndices)

        for i in range(self.numDrawingStuffIn3D - 1):
            glBindVertexArray(self.drawingStuffVAO[i])

            glBindBuffer(GL_ARRAY_BUFFER, self.drawingStuffVBO[i])
            glBufferData(GL_ARRAY_BUFFER, self.drawingStuffVerticesList[0].nbytes, self.drawingStuffVerticesList[0], GL_STATIC_DRAW)

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.drawingStuffEBO[i])
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.drawingStuffIndicesList[0].nbytes, self.drawingStuffIndicesList[0], GL_STATIC_DRAW)

            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.drawingStuffVerticesList[0].itemsize * self.numVertexComponentsWithTexCoord, ctypes.c_void_p(0))

            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, self.drawingStuffVerticesList[0].itemsize * self.numVertexComponentsWithTexCoord, ctypes.c_void_p(self.drawingStuffVerticesList[0].itemsize * 3))

            glEnableVertexAttribArray(2)
            glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, self.drawingStuffVerticesList[0].itemsize * self.numVertexComponentsWithTexCoord, ctypes.c_void_p(self.drawingStuffVerticesList[0].itemsize * 7))

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0)

    def _InitializeDrawingStuffIn2D(self):
        self.drawingStuffVAO = glGenVertexArrays(self.numDrawingStuffIn2D)
        self.drawingStuffVBO = glGenBuffers(self.numDrawingStuffIn2D)
        self.drawingStuffEBO = glGenBuffers(self.numDrawingStuffIn2D)

        intervalX = 50.0
        intervalY = 50.0

        screenPos = gSceneManager.GetScreenPos()

        triangleVerticesData = [
            screenPos[0][0] + intervalX, screenPos[0][1] + intervalY, 0.0, 1.0, 0.0, 0.0, 0.5,
            screenPos[1][0] - intervalX, screenPos[0][1] + intervalY, 0.0, 0.0, 1.0, 0.0, 0.5,
            (screenPos[0][0] + intervalX + screenPos[1][0] - intervalX ) / 2.0, screenPos[1][1] - intervalY, 0.0, 0.0, 0.0, 1.0, 0.5
            ]

        triangleIndicesData = [
            0, 1, 2
            ]        

        self.object1sVertices = np.array(triangleVerticesData, dtype = np.float32)
        self.object1sIndices = np.array(triangleIndicesData, dtype = np.uint32)

        self.drawingStuffVerticesList.clear()
        self.drawingStuffIndicesList.clear()

        self.drawingStuffVerticesList.append(self.object1sVertices)
        
        self.drawingStuffIndicesList.append(self.object1sIndices)        

        for i in range(self.numDrawingStuffIn2D - 1):
            glBindVertexArray(self.drawingStuffVAO[i])

            glBindBuffer(GL_ARRAY_BUFFER, self.drawingStuffVBO[i])
            glBufferData(GL_ARRAY_BUFFER, self.drawingStuffVerticesList[0].nbytes, self.drawingStuffVerticesList[0], GL_STATIC_DRAW)

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.drawingStuffEBO[i])
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.drawingStuffIndicesList[0].nbytes, self.drawingStuffIndicesList[0], GL_STATIC_DRAW)

            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.drawingStuffVerticesList[0].itemsize * self.numVertexComponents, ctypes.c_void_p(0))

            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, self.drawingStuffVerticesList[0].itemsize * self.numVertexComponents, ctypes.c_void_p(self.drawingStuffVerticesList[0].itemsize * 3))

            glEnableVertexAttribArray(2)
            glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, self.drawingStuffVerticesList[0].itemsize * self.numVertexComponents, ctypes.c_void_p(self.drawingStuffVerticesList[0].itemsize * 7))

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0)

    def _InitializeGUIStuff(self):
        gSceneManager.ClearSpecificProgramArgs()

        displaySize = gSceneManager.GetDisplaySize()
        screenPos = gSceneManager.GetScreenPos()

        backgroundVerticesData = [
            0.0, 0.0, 5.0, 1.0, 0.0, 0.0, 0.5,
            screenPos[0][0], 0.0, 5.0, 1.0, 0.0, 0.0, 0.5,
            screenPos[0][0], displaySize[1], 5.0, 1.0, 0.0, 0.0, 0.5,
            0.0, displaySize[1], 5.0, 1.0, 0.0, 0.0, 0.5,

            screenPos[1][0], 0.0, 5.0, 1.0, 0.0, 0.0, 0.5,
            displaySize[0], 0.0, 5.0, 1.0, 0.0, 0.0, 0.5,
            displaySize[0], displaySize[1], 5.0, 1.0, 0.0, 0.0, 0.5,
            screenPos[1][0], displaySize[1], 5.0, 1.0, 0.0, 0.0, 0.5,

            0.0, 0.0, 5.0, 1.0, 0.0, 0.0, 0.5,
            displaySize[0], 0.0, 5.0, 1.0, 0.0, 0.0, 0.5,
            displaySize[0], screenPos[0][1], 5.0, 1.0, 0.0, 0.0, 0.5,
            0.0, screenPos[0][1], 5.0, 1.0, 0.0, 0.0, 0.5,

            0.0, screenPos[1][1], 5.0, 1.0, 0.0, 0.0, 0.5,
            displaySize[0], screenPos[1][1], 5.0, 1.0, 0.0, 0.0, 0.5,
            displaySize[0], displaySize[1], 5.0, 1.0, 0.0, 0.0, 0.5,
            0.0, displaySize[1], 5.0, 1.0, 0.0, 0.0, 0.5
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
            0.0, screenPos[0][1], 8.0, 1.0, 1.0, 1.0, 1.0,
            displaySize[0], screenPos[0][1], 8.0, 1.0, 1.0, 1.0, 1.0,

            0.0, screenPos[1][1], 8.0, 1.0, 1.0, 1.0, 1.0,
            displaySize[0], screenPos[1][1], 8.0, 1.0, 1.0, 1.0, 1.0,

            screenPos[0][0], 0.0, 8.0, 1.0, 1.0, 1.0, 1.0,
            screenPos[0][0], displaySize[1], 8.0, 1.0, 1.0, 1.0, 1.0,

            screenPos[1][0], 0.0, 8.0, 1.0, 1.0, 1.0, 1.0,
            screenPos[1][0], displaySize[1], 8.0, 1.0, 1.0, 1.0, 1.0
            ]

        backgroundLineIndicesData = [
            0, 1,
            2, 3,
            4, 5,
            6, 7
            ]

        self.backgroundLineVertices = np.array(backgroundLineVerticesData, dtype = np.float32)
        self.backgroundLineIndices = np.array(backgroundLineIndicesData, dtype = np.uint32)
        
        self.GUIStuffVerticesList.clear()
        self.GUIStuffIndicesList.clear()

        self.GUIStuffVerticesList.append(self.backgroundVertices)        
        self.GUIStuffVerticesList.append(self.backgroundLineVertices)        

        self.GUIStuffIndicesList.append(self.backgroundIndices)        
        self.GUIStuffIndicesList.append(self.backgroundLineIndices)        

        for i in range(self.numGUIStuff):
            glBindVertexArray(self.GUIVAO[i])

            glBindBuffer(GL_ARRAY_BUFFER, self.GUIVBO[i])
            glBufferData(GL_ARRAY_BUFFER, self.GUIStuffVerticesList[i].nbytes, self.GUIStuffVerticesList[i], GL_STATIC_DRAW)

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.GUIEBO[i])
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.GUIStuffIndicesList[i].nbytes, self.GUIStuffIndicesList[i], GL_STATIC_DRAW)

            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.GUIStuffVerticesList[i].itemsize * self.numVertexComponents, ctypes.c_void_p(0))

            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, self.GUIStuffVerticesList[i].itemsize * self.numVertexComponents, ctypes.c_void_p(self.GUIStuffVerticesList[i].itemsize * 3))

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        gSceneManager.AddSpecificProgramArgs(gSceneManager.GetColor('DefaultColor_', 7), [685, 17], 0, 1, 'Medium', self.programName)

    def _Update3D(self, deltaTime):
        self.rotDegree += deltaTime * 50

        if self.rotDegree > 360.0:
            self.rotDegree = 0.0

    def _Update2D(self, deltaTime):
        pass

    def _DrawDrawingStuffIn3D(self):
        glPushAttrib(GL_COLOR_BUFFER_BIT | GL_ENABLE_BIT)

        glEnable(GL_BLEND)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        transMat = glm.translate(glm.vec3(0.0, 0.0, 0.0))
        
        rotXMat = glm.rotate(glm.radians(self.rotDegree), glm.vec3(1.0, 0.0, 0.0))
        rotYMat = glm.rotate(glm.radians(self.rotDegree), glm.vec3(0.0, 1.0, 0.0))
        rotZMat = glm.rotate(glm.radians(self.rotDegree), glm.vec3(0.0, 0.0, 1.0))       

        scaleMat = glm.scale(glm.vec3(1.0, 1.0, 1.0))

        self.drawingModelMat = transMat * rotZMat * rotYMat * rotXMat * scaleMat        

        shader = gSceneManager.GetShader()
        shader.SetMat4('modelMat', self.drawingModelMat)        

        glBindVertexArray(self.drawingStuffVAO[0])
        glDrawElements(GL_TRIANGLES, len(self.drawingStuffIndicesList[0]), GL_UNSIGNED_INT, None)

        glBindVertexArray(0)

        glPopAttrib()

    def _DrawDrawingStuffIn2D(self):
        glPushAttrib(GL_COLOR_BUFFER_BIT | GL_ENABLE_BIT)

        glEnable(GL_BLEND)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        shader = gSceneManager.GetShader()
        shader.SetMat4('modelMat', self.drawingModelMat)        

        glBindVertexArray(self.drawingStuffVAO[0])
        glDrawElements(GL_TRIANGLES, len(self.drawingStuffIndicesList[0]), GL_UNSIGNED_INT, None)

        glBindVertexArray(0)

        glPopAttrib()

    def _DrawGUI(self):
        glPushAttrib(GL_COLOR_BUFFER_BIT | GL_ENABLE_BIT | GL_LINE_BIT)

        glDisable(GL_DEPTH_TEST)

        glEnable(GL_BLEND)

        glBlendFunc(GL_SRC_ALPHA, GL_ZERO)

        shader = gSceneManager.GetShader()
        shader.SetMat4('modelMat', self.GUIModelMat)

        GUIStuffIndex = 0
        
        glBindVertexArray(self.GUIVAO[GUIStuffIndex])
        glDrawElements(GL_TRIANGLES, len(self.GUIStuffIndicesList[GUIStuffIndex]), GL_UNSIGNED_INT, None)        
        
        GUIStuffIndex += 1

        glLineWidth(2.0)

        glBindVertexArray(self.GUIVAO[GUIStuffIndex])
        glDrawElements(GL_LINES, len(self.GUIStuffIndicesList[GUIStuffIndex]), GL_UNSIGNED_INT, None)
        
        glBindVertexArray(0)

        glPopAttrib()

class Shader:
    def __init__(self, vsCode, fsCode):
        self.program = None

        self.program = compileProgram(compileShader(vsCode, GL_VERTEX_SHADER), compileShader(fsCode, GL_FRAGMENT_SHADER))

    def Use(self):
        glUseProgram(self.program)

    def SetBool(self, name, value):
        loc = glGetUniformLocation(self.program, name)
        
        glUniform1i(loc, value)

    def SetVec2(self, name, x, y):
        loc = glGetUniformLocation(self.program, name)
        
        glUniform2f(loc, x, y)

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

class Box:
    def __init__(self, pos, size):
        self.pos = pos
        self.size = size

    def GetPos(self):
        return self.pos

    def GetSize(self):
        return self.size

    def GenerateInnerBoxes(self):
        boxes = []
        checkSum = 0

        for z in range(-1, 2):
            for y in range(-1, 2):
                for x in range(-1, 2):
                    checkSum = abs(x) + abs(y) + abs(z)
                    innerBoxSize = self.size / 3.0

                    if checkSum > 1:
                        posX = self.pos.x + x * innerBoxSize
                        posY = self.pos.y + y * innerBoxSize
                        posZ = self.pos.z + z * innerBoxSize

                        boxes.append(Box(glm.vec3(posX, posY, posZ), innerBoxSize))

        return boxes

class MengerSpongeFractal:
    def __init__(self, programName):
        self.programName = programName

        self.drawingStuffVerticesList = []
        self.drawingStuffIndicesList = []

        self.cubeVertices = []
        self.cubeIndicesList = []

        self.GUIStuffVerticesList = []
        self.GUIStuffIndicesList = []

        self.backgroundVertices = []
        self.backgroundIndices = []

        self.backgroundLineVertices = []
        self.backgroundLineIndices = []

        self.rotDegree = 0.0
        self.rotSpeed = 20.0

        self.drawingModelMat = glm.mat4()
        self.GUIModelMat = glm.mat4()

        self.level = 0
        self.numBoxes = 1

        self.boxes = []

        self.numVertexComponents = 7        
        self.numDrawingStuff = 1
        self.numInstanceStuff = 1
        self.numGUIStuff = 2

        self.drawingStuffVAO = glGenVertexArrays(self.numDrawingStuff)
        self.drawingStuffVBO = glGenBuffers(self.numDrawingStuff)
        self.instanceStuffVBO = glGenBuffers(self.numInstanceStuff)
        self.drawingStuffEBO = glGenBuffers(self.numDrawingStuff)

        self.GUIVAO = glGenVertexArrays(self.numGUIStuff)
        self.GUIVBO = glGenBuffers(self.numGUIStuff)
        self.GUIEBO = glGenBuffers(self.numGUIStuff)

        self._Initialize()

    def Restart(self):
        self._Initialize()

    def UpdateAboutKeyInput(self, key, value = True):
        if key == glfw.KEY_3:
            self.level -= 1
            self.numBoxes = int(self.numBoxes / 20)

            if self.level < 0:
                self.level = 0
                self.numBoxes = 1

            self.Restart()

        elif key == glfw.KEY_4:
            self.level +=1
            self.numBoxes *= 20

            if self.level > 5:
                self.level = 5
                self.numBoxes = 3200000

            self.Restart()

    def UpdateAboutMouseInput(self, button, pos):
        pass

    def Update(self, deltaTime):
        self.rotDegree += deltaTime * self.rotSpeed

        if self.rotDegree > 360.0:
            self.rotDegree -= 360.0

    def Draw(self):
        shader = gSceneManager.GetShader()

        displaySize = gSceneManager.GetDisplaySize()
        screenPos = gSceneManager.GetScreenPos()
        screenSize = gSceneManager.GetScreenSize()

        shader.SetMat4('prjMat', gSceneManager.GetPerspectivePrjMat())
        shader.SetMat4('viewMat', gSceneManager.GetView3DMat())

        glViewport(screenPos[0][0], screenPos[0][1], screenSize[0], screenSize[1])

        self._DrawDrawingStuff()

        glViewport(0, 0, displaySize[0], displaySize[1])

        shader.SetMat4('prjMat', gSceneManager.GetOrthoPrjMat())
        shader.SetMat4('viewMat', gSceneManager.GetView2DMat())

        self._DrawGUI()

    def _InitializeCube(self):
        cubeVerticesData = [
            # Front
            -0.5, -0.5, 0.5, 1.0, 0.0, 0.0, 1.0, 
            0.5, -0.5, 0.5, 1.0, 0.0, 0.0, 1.0,
            0.5, 0.5, 0.5, 1.0, 0.0, 0.0, 1.0,
            -0.5, 0.5, 0.5, 1.0, 0.0, 0.0, 1.0,

            # Back
            0.5, -0.5, -0.5, 0.0, 1.0, 0.0, 1.0,
            -0.5, -0.5, -0.5, 0.0, 1.0, 0.0, 1.0,
            -0.5, 0.5, -0.5, 0.0, 1.0, 0.0, 1.0,
            0.5, 0.5, -0.5, 0.0, 1.0, 0.0, 1.0,

            # Left
            -0.5, -0.5, -0.5, 0.0, 0.0, 1.0, 1.0,
            -0.5, -0.5, 0.5, 0.0, 0.0, 1.0, 1.0,
            -0.5, 0.5, 0.5, 0.0, 0.0, 1.0, 1.0,
            -0.5, 0.5, -0.5, 0.0, 0.0, 1.0, 1.0,

            # Right
            0.5, -0.5, 0.5, 1.0, 1.0, 0.0, 1.0,
            0.5, -0.5, -0.5, 1.0, 1.0, 0.0, 1.0,
            0.5, 0.5, -0.5, 1.0, 1.0, 0.0, 1.0,
            0.5, 0.5, 0.5, 1.0, 1.0, 0.0, 1.0,

            # Top
            -0.5, 0.5, 0.5, 0.0, 1.0, 1.0, 1.0,
            0.5, 0.5, 0.5, 0.0, 1.0, 1.0, 1.0,
            0.5, 0.5, -0.5, 0.0, 1.0, 1.0, 1.0,
            -0.5, 0.5, -0.5, 0.0, 1.0, 1.0, 1.0,

            # Bottom
            -0.5, -0.5, -0.5, 1.0, 0.0, 1.0, 1.0,
            0.5, -0.5, -0.5, 1.0, 0.0, 1.0, 1.0,
            0.5, -0.5, 0.5, 1.0, 0.0, 1.0, 1.0,
            -0.5, -0.5, 0.5, 1.0, 0.0, 1.0, 1.0
            ]

        cubeIndicesData = [
            0, 1, 2, 2, 3, 0,
            4, 5, 6, 6, 7, 4,
            8, 9, 10, 10, 11, 8,
            12, 13, 14, 14, 15, 12,
            16, 17, 18, 18, 19, 16,
            20, 21, 22, 22, 23, 20
            ]

        return cubeVerticesData, cubeIndicesData

    def _InitializeInstance(self):
        instanceTotalModelMat = []

        for i in range(len(self.boxes)):
            boxPos = self.boxes[i].GetPos()
            boxSize = self.boxes[i].GetSize()

            transMat = glm.translate(glm.vec3(boxPos.x, boxPos.y, boxPos.z))
            scaleMat = glm.scale(glm.vec3(boxSize, boxSize, boxSize))

            instanceModelMat = transMat * scaleMat
            instanceTotalModelMat.append(np.transpose(np.array(instanceModelMat)))

        instanceTotalModelMat1D = np.array(instanceTotalModelMat, np.float32).flatten()

        instanceTotalModelMatLen = len(instanceTotalModelMat)
        instanceModelMatBytes = int(instanceTotalModelMat1D.nbytes / instanceTotalModelMatLen)
        instanceModelMatDivideBytes = int(instanceModelMatBytes / 4)

        glBindBuffer(GL_ARRAY_BUFFER, self.instanceStuffVBO)
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

    def _Initialize(self):
        gSceneManager.SetView3D(True)

        self._InitializeDrawingStuff()

        self._InitializeGUIStuff()

    def _InitializeDrawingStuff(self):
        gSceneManager.ClearSpecificProgramArgs()

        self.boxes.clear()
        self.boxes.append(Box(glm.vec3(0.0, 0.0, 0.0), 5.0))
        innerBoxes = []

        for i in range(self.level):
            for j in range(len(self.boxes)):
                subInnerBoxes = self.boxes[j].GenerateInnerBoxes()
                innerBoxes.extend(subInnerBoxes)

            self.boxes.clear()
            self.boxes = innerBoxes.copy()
            innerBoxes.clear()

        cubeVerticesData, cubeIndicesData = self._InitializeCube()

        self.cubeVertices = np.array(cubeVerticesData, dtype = np.float32)
        self.cubeIndices = np.array(cubeIndicesData, dtype = np.uint32)
        
        self.drawingStuffVerticesList.clear()
        self.drawingStuffIndicesList.clear()

        self.drawingStuffVerticesList.append(self.cubeVertices)

        self.drawingStuffIndicesList.append(self.cubeIndices)

        glBindVertexArray(self.drawingStuffVAO)

        glBindBuffer(GL_ARRAY_BUFFER, self.drawingStuffVBO)
        glBufferData(GL_ARRAY_BUFFER, self.cubeVertices.nbytes, self.cubeVertices, GL_STATIC_DRAW)
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.drawingStuffEBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.cubeIndices.nbytes, self.cubeIndices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.cubeVertices.itemsize * self.numVertexComponents, ctypes.c_void_p(0))

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, self.cubeVertices.itemsize * self.numVertexComponents, ctypes.c_void_p(self.cubeVertices.itemsize * 3))

        self._InitializeInstance()

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        levelStr = ".Level : " + str(self.level)
        numBoxesStr = ".  # Bixes : " + format(self.numBoxes, ',')

        gSceneManager.AddSpecificProgramArgs(gSceneManager.GetColor('DefaultColor_', 7), [585, 570], 0, 1, 'Large', levelStr)

        gSceneManager.AddSpecificProgramArgs(gSceneManager.GetColor('DefaultColor_', 7), [585, 535], 0, 1, 'Medium', numBoxesStr)

        gSceneManager.AddSpecificProgramArgs(gSceneManager.GetColor('DefaultColor_', 7), [625, 17], 0, 1, 'Medium', self.programName)

    def _InitializeGUIStuff(self):
        displaySize = gSceneManager.GetDisplaySize()
        screenPos = gSceneManager.GetScreenPos()

        backgroundVerticesData = [
            0.0, 0.0, 5.0, 1.0, 0.0, 0.0, 0.5,
            screenPos[0][0], 0.0, 5.0, 1.0, 0.0, 0.0, 0.5,
            screenPos[0][0], displaySize[1], 5.0, 1.0, 0.0, 0.0, 0.5,
            0.0, displaySize[1], 5.0, 1.0, 0.0, 0.0, 0.5,

            screenPos[1][0], 0.0, 5.0, 1.0, 0.0, 0.0, 0.5,
            displaySize[0], 0.0, 5.0, 1.0, 0.0, 0.0, 0.5,
            displaySize[0], displaySize[1], 5.0, 1.0, 0.0, 0.0, 0.5,
            screenPos[1][0], displaySize[1], 5.0, 1.0, 0.0, 0.0, 0.5,

            0.0, 0.0, 5.0, 1.0, 0.0, 0.0, 0.5,
            displaySize[0], 0.0, 5.0, 1.0, 0.0, 0.0, 0.5,
            displaySize[0], screenPos[0][1], 5.0, 1.0, 0.0, 0.0, 0.5,
            0.0, screenPos[0][1], 5.0, 1.0, 0.0, 0.0, 0.5,

            0.0, screenPos[1][1], 5.0, 1.0, 0.0, 0.0, 0.5,
            displaySize[0], screenPos[1][1], 5.0, 1.0, 0.0, 0.0, 0.5,
            displaySize[0], displaySize[1], 5.0, 1.0, 0.0, 0.0, 0.5,
            0.0, displaySize[1], 5.0, 1.0, 0.0, 0.0, 0.5
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
            0.0, screenPos[0][1], 8.0, 1.0, 1.0, 1.0, 1.0,
            displaySize[0], screenPos[0][1], 8.0, 1.0, 1.0, 1.0, 1.0,

            0.0, screenPos[1][1], 8.0, 1.0, 1.0, 1.0, 1.0,
            displaySize[0], screenPos[1][1], 8.0, 1.0, 1.0, 1.0, 1.0,

            screenPos[0][0], 0.0, 8.0, 1.0, 1.0, 1.0, 1.0,
            screenPos[0][0], displaySize[1], 8.0, 1.0, 1.0, 1.0, 1.0,

            screenPos[1][0], 0.0, 8.0, 1.0, 1.0, 1.0, 1.0,
            screenPos[1][0], displaySize[1], 8.0, 1.0, 1.0, 1.0, 1.0
            ]

        backgroundLineIndicesData = [
            0, 1,
            2, 3, 
            4, 5,
            6, 7
            ]

        self.backgroundLineVertices = np.array(backgroundLineVerticesData, dtype = np.float32)
        self.backgroundLineIndices = np.array(backgroundLineIndicesData, dtype = np.uint32)

        self.GUIStuffVerticesList.clear()
        self.GUIStuffIndicesList.clear()

        self.GUIStuffVerticesList.append(self.backgroundVertices)
        self.GUIStuffVerticesList.append(self.backgroundLineVertices)

        self.GUIStuffIndicesList.append(self.backgroundIndices)
        self.GUIStuffIndicesList.append(self.backgroundLineIndices)

        for i in range(self.numGUIStuff):
            glBindVertexArray(self.GUIVAO[i])

            glBindBuffer(GL_ARRAY_BUFFER, self.GUIVBO[i])
            glBufferData(GL_ARRAY_BUFFER, self.GUIStuffVerticesList[i].nbytes, self.GUIStuffVerticesList[i], GL_STATIC_DRAW)

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.GUIEBO[i])
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.GUIStuffIndicesList[i].nbytes, self.GUIStuffIndicesList[i], GL_STATIC_DRAW)

            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.GUIStuffVerticesList[i].itemsize * self.numVertexComponents, ctypes.c_void_p(0))

            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, self.GUIStuffVerticesList[i].itemsize * self.numVertexComponents, ctypes.c_void_p(self.GUIStuffVerticesList[i].itemsize * 3))

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        gSceneManager.AddSpecificProgramArgs(gSceneManager.GetColor('DefaultColor_', 7), [625, 17], 0, 1, 'Medium', self.programName)

    def _DrawDrawingStuff(self):
        glPushAttrib(GL_COLOR_BUFFER_BIT | GL_ENABLE_BIT)

        glEnable(GL_BLEND)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        transMat = glm.translate(glm.vec3(0.0, 0.0, 0.0))

        rotXMat = glm.rotate(glm.radians(self.rotDegree), glm.vec3(1.0, 0.0, 0.0))
        rotYMat = glm.rotate(glm.radians(self.rotDegree), glm.vec3(0.0, 1.0, 0.0))
        rotZMat = glm.rotate(glm.radians(self.rotDegree), glm.vec3(0.0, 0.0, 1.0))

        rotMat = rotZMat * rotYMat * rotXMat

        scaleMat = glm.scale(glm.vec3(1.0, 1.0, 1.0))

        self.drawingModelMat = transMat * rotMat * scaleMat

        shader = gSceneManager.GetShader()
        shader.SetBool('useInstance', True)
        shader.SetMat4('modelMat', self.drawingModelMat)

        glBindVertexArray(self.drawingStuffVAO)
        glDrawElementsInstanced(GL_TRIANGLES, len(self.cubeIndices), GL_UNSIGNED_INT, None, len(self.boxes))

        glBindVertexArray(0)

        glPopAttrib()

    def _DrawGUI(self):
        glPushAttrib(GL_COLOR_BUFFER_BIT | GL_ENABLE_BIT | GL_LINE_BIT)

        glDisable(GL_DEPTH_TEST)

        glEnable(GL_BLEND)

        glBlendFunc(GL_SRC_ALPHA, GL_ZERO)

        shader = gSceneManager.GetShader()
        shader.SetBool('useInstance', False)
        shader.SetMat4('modelMat', self.GUIModelMat)

        GUIStuffIndex = 0

        glBindVertexArray(self.GUIVAO[GUIStuffIndex])
        glDrawElements(GL_TRIANGLES, len(self.GUIStuffIndicesList[GUIStuffIndex]), GL_UNSIGNED_INT, None)

        GUIStuffIndex += 1

        glLineWidth(2.0)

        glBindVertexArray(self.GUIVAO[GUIStuffIndex])
        glDrawElements(GL_LINES, len(self.GUIStuffIndicesList[GUIStuffIndex]), GL_UNSIGNED_INT, None)

        glBindVertexArray(0)

        glPopAttrib()

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
        elif key == glfw.KEY_F:
            gInputManager.SetKeyState(glfw.KEY_F, True)
        elif key == glfw.KEY_G:
            gInputManager.SetKeyState(glfw.KEY_G, True)
        elif key == glfw.KEY_H:
            gInputManager.SetKeyState(glfw.KEY_H, True)
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
        elif key == glfw.KEY_Q:
            gInputManager.SetKeyState(glfw.KEY_Q, True)
        elif key == glfw.KEY_E:
            gInputManager.SetKeyState(glfw.KEY_E, True)

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
        elif key == glfw.KEY_Q:
            gInputManager.SetKeyState(glfw.KEY_Q, False)
        elif key == glfw.KEY_E:
            gInputManager.SetKeyState(glfw.KEY_E, False)

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
    
    screenPos = gSceneManager.GetScreenPos()

    if gSceneManager.GetAbleMouseEscape() == False:
        mouseCheckInterval = 20

        if xPos < screenPos[0][0] + mouseCheckInterval:
            glfw.set_cursor_pos(glfwWindow, screenPos[1][0] - mouseCheckInterval, yPos)
            gInputManager.SetMouseEntered(False)
        elif xPos > screenPos[1][0] - mouseCheckInterval:
            glfw.set_cursor_pos(glfwWindow, screenPos[0][0] + mouseCheckInterval, yPos)
            gInputManager.SetMouseEntered(False)

        if yPos < screenPos[0][1] + mouseCheckInterval:
            glfw.set_cursor_pos(glfwWindow, xPos, screenPos[1][1] - mouseCheckInterval)
            gInputManager.SetMouseEntered(False)
        elif yPos > screenPos[1][1] - mouseCheckInterval:
            glfw.set_cursor_pos(glfwWindow, xPos, screenPos[0][1] + mouseCheckInterval)
            gInputManager.SetMouseEntered(False)

    else:
        if xPos < screenPos[0][0] or screenPos[1][0] < xPos:
            gInputManager.SetMouseEntered(False)
        elif yPos < screenPos[0][1] or screenPos[1][1] < yPos:
            gInputManager.SetMouseEntered(False)

    gSceneManager.SetDirty(True)

def Main():
    displaySize = gSceneManager.GetDisplaySize()
    projectName = "Menger Sponge Fractal"
    programName = "# MengerSpongeFractal"

    if not glfw.init():
        return

    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)

    glfwWindow = glfw.create_window(displaySize[0], displaySize[1], projectName + '.Part 1', None, None)

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
    gSceneManager.AddObject(MengerSpongeFractal(programName))
    
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