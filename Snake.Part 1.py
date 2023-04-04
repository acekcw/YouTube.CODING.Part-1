
from OpenGL.GL import *
from OpenGL.GLU import *
from enum import Enum

import glfw
import math
import random
import numpy as np
import freetype as ft


DISPLAY_SIZE = (800, 600)

gRectangleData = (
    (-0.5, -0.5, 0.0),
    (0.5, 0.5, 0.0)
    )

gSnakeHeadData = (
    (-0.5, -0.5, 0.0),
    (0.5, -0.5, 0.0),
    (0.5, 0.0, 0.0),
    (0.25, 0.5, 0.0),
    (-0.25, 0.5, 0.0),
    (-0.5, 0.0, 0.0)
    )

class Color(Enum):
    WHITE = (1.0, 1.0, 1.0)
    BLACK = (0.0, 0.0, 0.0)
    RED = (1.0, 0.0, 0.0)
    GREEN = (0.0, 1.0, 0.0)    
    BLUE = (0.0, 0.0, 1.0)
    PAUSE = (0.8, 0.0, 0.0)
    FEED = (0.5, 0.0, 0.0)
    SNAKE_HEAD = (0.0, 0.6, 0.0)
    SNAKE_BODY = (0.0, 0.3, 0.0)
    SCORE = (0.6, 0.6, 0.0)

class GameManager:
    def __init__(self):
        self.objectsData = []
        # 0: Left, 1: Right, 2: Up, 3: Down
        self.move = [False, False, False, False]
        self.boundaryPos = [0, 0]

        self.fps = 15
        self.timePerFps = 1.0 / self.fps

        self.restart = False
        
        self._Initialize()

    def GetMove(self):
        numMoveDirs = len(self.move)

        for i in range(numMoveDirs):
            if self.move[i] == True:
                return i

        return -1

    def SetMove(self, index, value):
        if self.pause == True:
            return

        numMoveDirs = len(self.move)

        self.move = [False for i in range(numMoveDirs)]
        self.move[index] = value

    def GetObjectsData(self, objIndex, state):
        return self.objectsData[objIndex][state]

    def SetObjectsData(self, objIndex, state, value):
        self.objectsData[objIndex][state] = value

    def GetScore(self):
        return self.score

    def SetScore(self, value):
        self.score = value

    def GetTimePerFps(self):
        return self.timePerFps

    def GetBoundaryPos(self):
        return self.boundaryPos

    def GetPause(self):
        return self.pause

    def GetRestart(self):
        return self.restart

    def SetNumObjects(self, *args):
        for numObjects in args:
            self.objectsData.append([numObjects, False])
            
    def UpdateGame(self):
        if self.pause == True:
            return

        if not self.dirty == True:
            return

        #print('GameManager Dirty')

        modelViewMat = np.array(glGetFloatv(GL_MODELVIEW_MATRIX))
        projectionMat = np.array(glGetFloatv(GL_PROJECTION_MATRIX))

        modelViewPrjMat = np.matmul(modelViewMat, projectionMat)
        invModelViewPrjMat = np.linalg.inv(modelViewPrjMat)

        cp = [0.0, 0.0, 0.0, 1.0]
        cpNdc = np.matmul(cp, modelViewPrjMat)
        cpNdc /= cpNdc[3]

        bpNdc = [1.0, 1.0, cpNdc[2], 1.0]
        bp = np.matmul(bpNdc, invModelViewPrjMat)
        bp /= bp[3]

        self.boundaryPos[0] = math.floor(bp[0])
        self.boundaryPos[1] = math.floor(bp[1])

        #print('boundaryPos: {0}'.format(self.boundaryPos))

        self.dirty = False

    def PauseGame(self):
        self.pause = not self.pause

    def RestartGame(self):
        self.restart = True

        self._Initialize()

    def CheckCompleted(self):
        if self.restart == True:
            self._CheckRestartCompleted()

    def _Initialize(self):
        numMoveDirs = len(self.move)
        self.move = [False for i in range(numMoveDirs)]

        numObjectsData = len(self.objectsData)
        numObjects = []

        for i in range(numObjectsData):
            numObjects.append(self.objectsData[i][0])

        self.objectsData.clear()

        for i in range(numObjectsData):
            self.objectsData.append([numObjects[i], False])

        self.score = 0          

        self.pause = False
        self.dirty = True

    def _CheckRestartCompleted(self):
        numObjectsData = len(self.objectsData)

        for i in range(numObjectsData):
            if self.objectsData[i][1] == False:
                return

        self.restart = False

class Snake:
    def __init__(self, gameManager):
        self.headVertices = []
        self.headIndices = []
        self.headBBVertices = []

        self.bodyPos = []
        self.bodyVertices = []
        self.bodyIndices = []
        
        self.partBodySize = 1
        
        self.headIndices.append(0)
        self.headIndices.append(1)
        self.headIndices.append(2)
        self.headIndices.append(3)
        self.headIndices.append(4)
        self.headIndices.append(5)

        self._Initialize(gameManager)

    def GetHeadPos(self):
        return self.headPos

    def GetHeadVertices(self):
        return self.headVertices

    def GetHeadIndices(self):
        return self.headIndices

    def GetHeadBBVertices(self):
        return self.headBBVertices

    def GetBodyVertices(self):
        return self.bodyVertices

    def GetBodyIndices(self):
        return self.bodyIndices

    def Update(self, gameManager):
        if gameManager.GetRestart() == True:
            self._Initialize(gameManager)

        if gameManager.GetPause() == True:
            return

        moveDir = gameManager.GetMove()

        #print('headPos: {0}'.format(self.headPos))

        if not moveDir == -1:
            self.dirty = True

        if not self.dirty == True:
            return

        #print('Snake Dirty')

        self.bodyPos.append([self.headPos[0], self.headPos[1], self.headPos[2]])

        if moveDir == 0:
            self.headPos[0] -= self.partBodySize
        elif moveDir == 1:
            self.headPos[0] += self.partBodySize
        elif moveDir == 2:
            self.headPos[1] += self.partBodySize
        elif moveDir == 3:
            self.headPos[1] -= self.partBodySize

        if len(self.bodyPos) > self.bodySize:
            del self.bodyPos[0]           

        self._GenerateHeadVertices(moveDir)

        self._GenerateBodyVerticesNIndices()

        self.dirty = False

    def AddBody(self, value):
        self.bodySize += value

        self.dirty = True

    def CheckPenetrated(self):
        for bp in self.bodyPos:
            if bp == self.headPos:
                return True

        return False

    def _Initialize(self, gameManager):
        self.bodyPos.clear()
        
        self.headPos = [0, 0, 0]

        self.bodySize = 0

        self.dirty = True

        gameManager.SetObjectsData(0, 1, True)

    def _GenerateHeadVertices(self, moveDir):
        bbLbX = gRectangleData[0][0] * self.partBodySize + self.headPos[0]
        bbLbY = gRectangleData[0][1] * self.partBodySize + self.headPos[1]
        bbRtX = gRectangleData[1][0] * self.partBodySize + self.headPos[0]
        bbRtY = gRectangleData[1][1] * self.partBodySize + self.headPos[1]

        self.headBBVertices.clear()

        self.headBBVertices.append([bbLbX, bbLbY, 0.0])
        self.headBBVertices.append([bbRtX, bbLbY, 0.0])
        self.headBBVertices.append([bbRtX, bbRtY, 0.0])
        self.headBBVertices.append([bbLbX, bbRtY, 0.0])

        RADIAN = math.pi / 180.0

        rotateAngle, rotateRadian = 0.0, 0.0

        if moveDir == 0:
            rotateAngle = 90.0
        elif moveDir == 1:
            rotateAngle = 270.0
        elif moveDir == 2:
            rotateAngle = 0.0
        elif moveDir == 3:
            rotateAngle = 180.0

        rotateRadian = rotateAngle * RADIAN

        tmpX, tmpY, pX, pY = 0.0, 0.0, 0.0, 0.0
        numSnakeHeadPoints = len(gSnakeHeadData)

        self.headVertices.clear()

        for i in range(numSnakeHeadPoints):
            tmpX = gSnakeHeadData[i][0] * self.partBodySize
            tmpY = gSnakeHeadData[i][1] * self.partBodySize

            pX = math.cos(rotateRadian) * tmpX - math.sin(rotateRadian) * tmpY
            pY = math.sin(rotateRadian) * tmpX + math.cos(rotateRadian) * tmpY

            pX += self.headPos[0]
            pY += self.headPos[1]

            self.headVertices.append([pX, pY, 0.0])

    def _GenerateBodyVerticesNIndices(self):
        lbX, lbY, rtX, rtY = 0.0, 0.0, 0.0, 0.0
        indexOffset = 0

        self.bodyVertices.clear()
        self.bodyIndices.clear()

        for i in range(self.bodySize):
            lbX = gRectangleData[0][0] * self.partBodySize + self.bodyPos[i][0]
            lbY = gRectangleData[0][1] * self.partBodySize + self.bodyPos[i][1]
            rtX = gRectangleData[1][0] * self.partBodySize + self.bodyPos[i][0]
            rtY = gRectangleData[1][1] * self.partBodySize + self.bodyPos[i][1]

            self.bodyVertices.append([lbX, lbY, 0.0])
            self.bodyVertices.append([rtX, lbY, 0.0])
            self.bodyVertices.append([rtX, rtY, 0.0])
            self.bodyVertices.append([lbX, rtY, 0.0])

            indexOffset = i * 4

            self.bodyIndices.append(indexOffset + 0)
            self.bodyIndices.append(indexOffset + 1)
            self.bodyIndices.append(indexOffset + 2)
            self.bodyIndices.append(indexOffset + 0)
            self.bodyIndices.append(indexOffset + 2)
            self.bodyIndices.append(indexOffset + 3)

class Feed:
    def __init__(self, gameManager):
        self.vertices = []
        self.indices = []

        self.size = 2

        self.indices.append(0)
        self.indices.append(1)
        self.indices.append(2)
        self.indices.append(0)
        self.indices.append(2)
        self.indices.append(3)
        
        self._Initialize(gameManager)
       
    def GetVertices(self):
        return self.vertices

    def GetIndices(self):
        return self.indices

    def SetDirty(self, value):
        self.dirty = value

    def Update(self, gameManager):
        if gameManager.GetRestart() == True:
            self._Initialize(gameManager)

        if gameManager.GetPause() == True:
            return

        if not self.dirty == True:
            return

        #print('Feed Dirty')

        boundaryPos = gameManager.GetBoundaryPos()

        self._GenerateRandPosNVertices(boundaryPos)

        self.dirty = False

    def _Initialize(self, gameManager):
        self.vertices.clear()

        self.pos = [0, 0, 0]

        self.dirty = True

        gameManager.SetObjectsData(1, 1, True)

    def _GenerateRandPosNVertices(self, boundaryPos):
        self.pos[0] = random.randrange(-boundaryPos[0], boundaryPos[0] + 1)
        self.pos[1] = random.randrange(-boundaryPos[1], boundaryPos[1] + 1)

        lbX = gRectangleData[0][0] * self.size + self.pos[0]
        lbY = gRectangleData[0][1] * self.size + self.pos[1]
        rtX = gRectangleData[1][0] * self.size + self.pos[0]
        rtY = gRectangleData[1][1] * self.size + self.pos[1]

        self.vertices.clear()

        self.vertices.append([lbX, lbY, 0.0])
        self.vertices.append([rtX, lbY, 0.0])
        self.vertices.append([rtX, rtY, 0.0])
        self.vertices.append([lbX, rtY, 0.0])

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

    def SetTexId(self, texId):
        self.texId = texId

    def GetListOffset(self):
        return self.listOffset

    def SetListOffset(self, listStartIndex):
        self.listOffset = listStartIndex - self.charStartOffset

    def GetBitmapData(self):
        return self.bitmapData

    def GetCharsSize(self):
        return self.charsSize

    def GetCharsAdvanceX(self):
        return self.charsAdvanceX

    def GetMaxCharHeight(self):
        return self.maxCharHeight

gGameManager = GameManager()


def HandleKeyCallback(glfwWindow, key, scanCode, action, modes):
    if action == glfw.PRESS:
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(glfwWindow, glfw.TRUE)
        if key == glfw.KEY_P:
            gGameManager.PauseGame()
        if key == glfw.KEY_R:
            gGameManager.RestartGame()

        if key == glfw.KEY_LEFT:
            gGameManager.SetMove(0, True)
        elif key == glfw.KEY_RIGHT:
            gGameManager.SetMove(1, True)
        elif key == glfw.KEY_UP:
            gGameManager.SetMove(2, True)
        elif key == glfw.KEY_DOWN:
            gGameManager.SetMove(3, True)

    #if action == glfw.RELEASE:
    #    if key == glfw.KEY_LEFT:
    #        gGameManager.SetMove(0, False)
    #    elif key == glfw.KEY_RIGHT:
    #        gGameManager.SetMove(1, False)
    #    elif key == glfw.KEY_UP:
    #        gGameManager.SetMove(2, False)
    #    elif key == glfw.KEY_DOWN:
    #        gGameManager.SetMove(3, False)

def MakeFontTex(font):
    bitmapData = font.GetBitmapData()
    charsSize = font.GetCharsSize()
    maxCharHeight = font.GetMaxCharHeight()

    texId = glGenTextures(1)

    glBindTexture(GL_TEXTURE_2D, texId)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)

    bitmapData = np.flipud(bitmapData)

    glTexImage2D(GL_TEXTURE_2D, 0, GL_ALPHA, bitmapData.shape[1], bitmapData.shape[0], 0,
                 GL_ALPHA, GL_UNSIGNED_BYTE, bitmapData)

    font.SetTexId(texId)

    dx = 0.0
    dy = maxCharHeight / float(bitmapData.shape[0])

    charsAdvanceX = font.GetCharsAdvanceX()

    listStartIndex = glGenLists(charsSize[0] * charsSize[1])
    font.SetListOffset(listStartIndex)

    for r in range(charsSize[0]):
        for c in range(charsSize[1]):
            glNewList(listStartIndex + r * charsSize[1] + c, GL_COMPILE)

            charIndex = r * charsSize[1] + c

            advanceX = charsAdvanceX[charIndex]
            dAdvanceX = advanceX / float(bitmapData.shape[1])

            glBegin(GL_QUADS)
            glTexCoord2f(dx, 1.0 - r * dy), glVertex3f(0.0, 0.0, 0.0)
            glTexCoord2f(dx + dAdvanceX, 1.0 - r * dy), glVertex3f(advanceX, 0.0, 0.0)
            glTexCoord2f(dx + dAdvanceX, 1.0 - (r + 1) * dy), glVertex3f(advanceX, -maxCharHeight, 0.0)
            glTexCoord2f(dx, 1.0 - (r + 1) * dy), glVertex3f(0.0, -maxCharHeight, 0.0)
            glEnd()

            glTranslatef(advanceX, 0.0, 0.0)

            glEndList()

            dx += dAdvanceX

        glTranslatef(0.0, -maxCharHeight, 0.0)
        dx = 0.0

def DrawSnakes(snakes):
    headVertices, headIndices = [], []
    bodyVertices, bodyIndices = [], []
    numHeadIndices, numBodyIndices = 0, 0

    numSnakes = gGameManager.GetObjectsData(0, 0)

    glEnableClientState(GL_VERTEX_ARRAY)

    for i in range(numSnakes):
        snakes[i].Update(gGameManager)

        headVertices = snakes[i].GetHeadVertices()
        headIndices = snakes[i].GetHeadIndices()
        bodyVertices = snakes[i].GetBodyVertices()
        bodyIndices = snakes[i].GetBodyIndices()

        numHeadIndices = len(headIndices)
        numBodyIndices = len(bodyIndices)

        glColor(Color.SNAKE_HEAD.value)

        glVertexPointer(3, GL_FLOAT, 0, headVertices)
        glDrawElements(GL_POLYGON, numHeadIndices, GL_UNSIGNED_INT, headIndices)

        glColor(Color.SNAKE_BODY.value)

        glVertexPointer(3, GL_FLOAT, 0, bodyVertices)
        glDrawElements(GL_TRIANGLES, numBodyIndices, GL_UNSIGNED_INT, bodyIndices)

    glDisableClientState(GL_VERTEX_ARRAY)

def DrawFeeds(feeds):
    feedVertices, feedIndices = [], []
    numFeedIndices = 0

    numFeeds = gGameManager.GetObjectsData(1, 0)

    glEnableClientState(GL_VERTEX_ARRAY)

    for i in range(numFeeds):
        feeds[i].Update(gGameManager)

        feedVertices = feeds[i].GetVertices()
        feedIndices = feeds[i].GetIndices()

        numFeedIndices = len(feedIndices)

        glColor(Color.FEED.value)

        glVertexPointer(3, GL_FLOAT, 0, feedVertices)
        glDrawElements(GL_TRIANGLES, numFeedIndices, GL_UNSIGNED_INT, feedIndices)

    glDisableClientState(GL_VERTEX_ARRAY)

def DrawText(font, x, y, text, color):
    glPushAttrib(GL_COLOR_BUFFER_BIT | GL_ENABLE_BIT)

    glEnable(GL_TEXTURE_2D)
    glEnable(GL_BLEND)

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    glMatrixMode(GL_PROJECTION)
    glPushMatrix()

    glLoadIdentity()

    glOrtho(0, DISPLAY_SIZE[0], 0, DISPLAY_SIZE[1], -1.0, 1.0)

    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()

    glLoadIdentity()

    glTranslatef(x, y, 0.0)

    texId = font.GetTexId()

    glBindTexture(GL_TEXTURE_2D, texId)

    glColor(color)

    glListBase(font.GetListOffset())
    glCallLists([ord(c) for c in text])

    #glBegin(GL_QUADS)
    #glTexCoord2f(0.0, 0.0), glVertex3f(0.0, 0.0, 0.0)
    #glTexCoord2f(1.0, 0.0), glVertex3f(400.0, 0.0, 0.0)
    #glTexCoord2f(1.0, 1.0), glVertex3f(400.0, 100.0, 0.0)
    #glTexCoord2f(0.0, 1.0), glVertex3f(0.0, 100.0, 0.0)
    #glEnd()

    glPopMatrix()

    glMatrixMode(GL_PROJECTION)
    glPopMatrix()

    glPopAttrib()

def EatFeeds(snakes, feeds):
    headBBVertices, feedVertices = [], []

    numSnakes = gGameManager.GetObjectsData(0, 0)
    numFeeds = gGameManager.GetObjectsData(1, 0)

    for i in range(numSnakes):
        headBBVertices = snakes[i].GetHeadBBVertices()

        for j in range(numFeeds):
            feedVertices = feeds[j].GetVertices()

            if headBBVertices[0][0] <= feedVertices[2][0] and headBBVertices[2][0] >= feedVertices[0][0]:
                if headBBVertices[0][1] <= feedVertices[2][1] and headBBVertices[2][1] >= feedVertices[0][1]:
                    snakes[i].AddBody(1)
                    feeds[i].SetDirty(True)

                    gGameManager.SetScore(gGameManager.GetScore() + 1)

def CheckGameOver(snakes):
    headPos = []

    numSnakes = gGameManager.GetObjectsData(0, 0)
    boundaryPos = gGameManager.GetBoundaryPos()

    for i in range(numSnakes):
        if snakes[i].CheckPenetrated() == True:
            gGameManager.RestartGame()

        headPos = snakes[i].GetHeadPos()

        if headPos[0] < -boundaryPos[0] or boundaryPos[0] < headPos[0]:
            gGameManager.RestartGame()
        elif headPos[1] < -boundaryPos[1] or boundaryPos[1] < headPos[1]:
            gGameManager.RestartGame()

def Main():
    snakes = []
    feeds = []

    numSnakes = 1
    numFeeds = 1

    gGameManager.SetNumObjects(numSnakes, numFeeds)

    snakes = [Snake(gGameManager) for i in range(numSnakes)]
    feeds = [Feed(gGameManager) for i in range(numFeeds)]

    smallFont = Font('..\Fonts\comic.ttf', 24)
    font = Font('..\Fonts\comic.ttf', 32)
    largeFont = Font('..\Fonts\comic.ttf', 128)

    #font = Font('..\Fonts\cour.ttf', 16)

    fovy = 45.0
    aspect = DISPLAY_SIZE[0] / DISPLAY_SIZE[1]
    near = 0.1
    far = 1000.0

    if not glfw.init():
        return

    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)

    glfwWindow = glfw.create_window(DISPLAY_SIZE[0], DISPLAY_SIZE[1], 'Snake.Part 1', None, None)

    if not glfwWindow:
        glfw.terminate()
        return

    videoMode = glfw.get_video_mode(glfw.get_primary_monitor())

    windowWidth = videoMode.size.width
    windowHeight = videoMode.size.height
    windowPosX = int(windowWidth / 2 - DISPLAY_SIZE[0] / 2) - 250
    windowPosY = int(windowHeight / 2 - DISPLAY_SIZE[1] / 2) - 50

    glfw.set_window_pos(glfwWindow, windowPosX, windowPosY)

    glfw.show_window(glfwWindow)

    glfw.make_context_current(glfwWindow)

    glfw.set_key_callback(glfwWindow, HandleKeyCallback)

    MakeFontTex(smallFont)
    MakeFontTex(font)
    MakeFontTex(largeFont)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    gluPerspective(fovy, aspect, near, far)

    glClearColor(Color.BLACK.value[0], Color.BLACK.value[1], Color.BLACK.value[2], 1.0)

    lastElapsedTime = glfw.get_time()

    while not glfw.window_should_close(glfwWindow):
        glfw.poll_events()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glTranslatef(0.0, 0.0, -40.0)

        gGameManager.UpdateGame()

        DrawSnakes(snakes)
        DrawFeeds(feeds)

        DrawText(smallFont, 660, 590, 'Score: {0}'.format(gGameManager.GetScore()), Color.SCORE.value)

        #if gGameManager.GetPause() == True:
        #    DrawText(largeFont, 170, 380, 'Pause', Color.PAUSE.value)

        glfw.swap_buffers(glfwWindow)

        EatFeeds(snakes, feeds)

        CheckGameOver(snakes)

        gGameManager.CheckCompleted()

        while glfw.get_time() < lastElapsedTime + gGameManager.GetTimePerFps():
            pass        

        lastElapsedTime = glfw.get_time()


glfw.terminate()


if __name__ == "__main__":
    Main()