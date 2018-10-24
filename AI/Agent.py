#!/usr/bin/env python3
# Agent.py
##written by Yijiang Wu z5121109
##firstly, we build a Map class to record the map by the 5*5 sight received by the TCPIP socket.
##The Map class has a coordinate system to record the pos(x,y) and the facing to record the agent facing.
##it has many functions to record the title tpye of the map. 'W' means not explore yet.transferMaptoNorth is to keep the map we record in a same direction.
##printMap() function is to print the map we recorded.

##secondly,we write an Agent class to find the treasure and get back.An agent has 4 item situation. whether he has a dynamite, raft,axe or key and we use self.inthewater to check
##if he is in the water.The first thing we do is check whether the agent can see the treasure. If not, then explore arround without raft by using explore function.in that function,
##the target position is the boundry of the map which is near the 'W' type. use markResource function to record the tools position. explore all the area the agent can walk until
##no walkable area without using raft and dynamite because each of it can just use once.Then check if there is a treasure in the map we recorded.If yes, the find the path to
##treasure by findsolution function.It traversal all possible move with different situation(position, key, axe, dynamite, raft, inwater) and produce a path to processStep function
##to send the command by step function.If the agent can not see the treasure in the map we record or there is no solution to get the treasure then try to explore on the water if
##agent can.Use exploreagain function to let the agent go to the water and once the agent on the water, he shouldn't get back to ground until he explore all the area on water.
##So, we make the tranfergound function in Map class to tranfer the ground type ' ' to 'g' which is not walkable for agent.When the agent finish exploreonwater function, we
##use transferback function to transfer the 'g' type to ' '.And then do the same thing as last step to check if there is a treasure in the map we recorded,but has a bit difference
##on the findsolution which is using removeWrongDynamite function to remove the wrong choice the agent use dynamite to blast the wall and findreturnPath function is to check
##whether it can go back, if yes, return path, if not , find another way.
##once the agent get the treasure,use findsolution function with the target position is (0,0) which is the oringinal position to get back.

##the kernel algorithm is to traversal all possible move.not only for position but also include the agent state for the tools and if he is in the water.if it is exactly the same
##move, then 'continue' for the while loop until if traversal all the possibilities.

import socket, argparse
NORTH = 0
EAST  = 1
SOUTH = 2
WEST  = 3
SIGHT = 5
X = 0
Y = 1

class Map:
    def __init__(self):
        self.ground = dict()
        self.facing = NORTH
        self.treasurepos = None
        self.pos = (0, 0)

    def __getitem__(self, item):
        return self.ground[item]

    def __setitem__(self, key, value):
        self.ground[key] = value

    def __contains__(self, item):
        return item in self.ground

    def __iter__(self):
        for x in self.ground:
            yield x

    def addType(self, pos, type):
        self[pos] = type

        northtitle = Map.north(pos)
        easttitle = Map.east(pos)
        southtitle = Map.south(pos)
        westtitle = Map.west(pos)
        if northtitle not in self:
            self[northtitle] = 'W'

        if easttitle not in self:
            self[easttitle] = 'W'

        if southtitle not in self:
            self[southtitle] = 'W'

        if westtitle not in self:
            self[westtitle] = 'W'

        if type == '^':
            self.setType(pos, ' ')

        if type == '$':
            self.treasurepos = pos

    def copy(self):
        newMap = Map()
        newMap.ground = self.ground.copy()
        return newMap

    def printMap(self):
        xlist, ylist = zip(*self)
        xmin,xmax,ymin,ymax = min(xlist), max(xlist), min(ylist), max(ylist)
        ww = ''.join([str(x%10) for x in range(xmin-1, xmax+1)])
        for i in range(ymin, ymax+1):
            ww += '\n'+str(i%10)
            for j in range(xmin, xmax+1):
                if (j, i) in self:
                    if self.myposition() == (j, i):
                        direction = {NORTH:'^', EAST:'>', SOUTH:'v', WEST:'<'}
                        ww += direction[self.getFacing()]
                    else:
                        ww += self[(j,i)]
                else:
                    ww += '?'
        return ww

    def setType(self, pos, type):
        self[pos] = type

    def getType(self, pos):
        return self.ground[pos]

    def myposition(self):
        return self.pos

    def setPos(self, pos):
        self.pos = pos

    def setFacing(self, facing):
        self.facing = facing % 4

    def getFacing(self):
        return self.facing

    def mapScope(self, scope):
        turnedMap = self.transferMaptoNorth(scope)
        x, y = self.myposition()
        for i in range(SIGHT):
            for j in range(SIGHT):
                self.addType((x + j - SIGHT//2, y + i - SIGHT//2), turnedMap[i][j])

    def transferMaptoNorth(self, scope):
        facing = self.facing
        newscope = scope
        if facing == EAST:
            newscope = self.spinMap([x[::-1] for x in newscope][::-1])
        elif facing == SOUTH:
            newscope = [x[::-1] for x in newscope][::-1]
        elif facing == WEST:
            newscope = self.spinMap(newscope)
        return newscope

    def spinMap(self, map):
        return tuple(zip(*map))[::-1]

    def hastreasure(self):
        return self.treasurepos != None

    def treasurePos(self):
        return self.treasurepos
            
    def resourcePos(self):
        return [x for x in self if self[x] in 'kad']

    def transferground(self):
        for x in self:
            if x == self.pos:
                continue
            if self[x] == ' ':
                self.setType(x, 'g')
    def transferback(self):
        for x in self:
            if x == self.pos:
                continue
            if self[x] == 'g':
                self.setType(x, ' ')
    def dynamitesOnGround(self):
        return len([x for x in self if self[x] == 'd'])

    def surroundingType(self, pos):
        x, y = pos
        return set([self[(i,j)] for i in range(x-SIGHT//2, x+SIGHT//2 +1)
                                for j in range(y-SIGHT//2, y+SIGHT//2 +1)
                                if (i,j) in self])


    def north(pos):
        return (pos[X], pos[Y] - 1)

    def east(pos):
        return (pos[X] + 1, pos[Y])

    def south(pos):
        return (pos[X], pos[Y]+1)

    def west(pos):
        return (pos[X] - 1, pos[Y])

    def distance(pos1, pos2):
        return abs(pos1[X] - pos2[X]) + abs(pos1[Y] - pos2[Y])

    def directionDiff(originPos, targetPos, facing):
        targetd = {( 0, -1): NORTH,
                   ( 1,  0): EAST,
                   ( 0,  1): SOUTH,
                   (-1,  0): WEST}
        return targetd[(targetPos[X]-originPos[X], targetPos[Y]-originPos[Y])] - facing

class Agent:
    SIGHT = 5
    def __init__(self, port):
        self.map = Map()
        self.pipe = Pipe(port, self.map)
        self.hasdynamite = 0
        self.hasraft = False
        self.hasAxe = False
        self.hasKey = False
        self.inthewater = False
        self.explore_again = False
        self.swimming = False
        self.end = False
        self.getback = False
    def begin(self):
        while(True):
            
            if self.explore_again == False:
                if self.map.hastreasure():
                    solutionPath = self.findsolution(self.map.myposition(), self.map.treasurePos(), self.hasKey, self.hasAxe, self.hasdynamite, self.hasraft)
                    if solutionPath:
                        self.processStep(solutionPath)
                        self.getback = True
                        returnPath = self.findsolution(self.map.myposition(), (0,0), self.hasKey, self.hasAxe, self.hasdynamite, self.hasraft)
                        self.processStep(returnPath)
                        
                        break

            if self.explore():
                while(self.explore()):
                    if self.markResource():
                        break
            else:
                self.inthewater = False
                self.explore_again = True
                
                if self.exploreagain():
                    
                    while(self.exploreagain()):
                        if self.markResource():
                            break
                else:
                    if self.swimming:
                        self.map.transferground()
                        if self.end == True:
                            self.swimming = False
                            self.explore_again = False
                            self.inthewater = True
                            self.map.transferback()
                            if self.explorefinal():                  
                                while(self.explorefinal()):
                                    if self.markResource():
                                        break
                            
                            if self.map.hastreasure():
                                self.map.setType(self.map.pos, '~')
                                solutionPath = self.findsolution(self.map.myposition(), self.map.treasurePos(), self.hasKey, self.hasAxe, self.hasdynamite, True)
                                if solutionPath:
                                    self.processStep2(solutionPath)
                                    self.getback = True
                                    returnPath = self.findsolution(self.map.myposition(), (0,0), self.hasKey, self.hasAxe, self.hasdynamite, self.hasraft)
                                    self.processStep(returnPath)
                                    break
                                else:
                                    break
                            
                        if self.exploreonwater():                        
                            while(self.exploreonwater()):
                                if self.markResource():
                                    break
                        else:
                            self.end = True
                    else:
                        break
                

    def markResource(self):
        resources = self.map.resourcePos()
        if resources:
            pathes = self.findPaths(resources)
            for path in pathes:
                self.processStep(path)
        return len(resources) != 0
##explore on water
    def processPathonwater(self, originPos, targetPos, key, axe, dynamite, raft):
        map = self.map
        possiblemove = [([originPos], key, axe, dynamite, raft, map)]
        went = set()
        while(possiblemove):
            path, key, axe, dynamite, raft, map = possiblemove[0]
            possiblemove = possiblemove[1:]
            titlepos = path[-1]
            if titlepos == targetPos:
                return path
            canwalk = {'~',' '}
            titleType = map.getType(titlepos)
            myMap = map
            if titleType not in canwalk:
                continue
            if titlepos in went:
                continue
            went.add(titlepos)
            
            distancetopos = [Map.north(titlepos), Map.south(titlepos), Map.east(titlepos), Map.west(titlepos)]
            distancetopos.sort(key=lambda x:Map.distance(targetPos, x),reverse=True)
            for pos in distancetopos:
                possiblemove.append((path+[pos], key, axe, dynamite,raft, myMap))        
        return []           
##traversal all possibilities and find the solution path            
    def findsolution(self, originPos, targetPos, key, axe, dynamite, raft):

        if (self.getback == False) and (self.inthewater == False):
            map = self.removeWrongDynamite(self.map, key, axe, dynamite, raft)

        else:
            map = self.map
        

        inwater = False
        if self.inthewater == True:
            inwater = True
        
        possiblemove = [([originPos], key, axe, dynamite, raft,inwater, map)]

        went = set()
        while(possiblemove):
            path, key, axe, dynamite, raft,inwater, map = possiblemove[0]
            possiblemove = possiblemove[1:]
            titlepos = path[-1]
            
            canwalk = {' ', 'k', 'a', 'd','$'}
            if key:
                canwalk.add('-')
            if axe:
                canwalk.add('T')
            if dynamite:
                canwalk.add('*')
            if raft:
                canwalk.add('~')
            
            titleType = map.getType(titlepos)
            if titlepos == targetPos:
                if titleType not in canwalk:
                    if titleType == '~':
                        continue

                if targetPos == self.map.treasurePos():

                    returnPath = self.findreturnPath(myMap1,titlepos, key, axe, dynamite, raft)
                    if returnPath:
                        return path
                    else:
                        continue
                else:
                    return path
                
            if titleType not in canwalk:
                continue
            
                
            if (titleType == '~') and (self.explore_again == True):

                self.swimming = True
                
            if (titleType == ' ') and (inwater == True):
                raft = False
                inwater = False

            if titleType in 'kda~T*':
                myMap = map.copy()
                myMap1 = map.copy()
                if titleType == 'k':
                    key = True
                    myMap.setType(titlepos, ' ')
                if titleType == 'd':
                    dynamite += 1
                    myMap.setType(titlepos, ' ')
                if titleType == 'a':
                    axe = True
                    myMap.setType(titlepos, ' ')

                if titleType == '~':
                    myMap.setType(titlepos,'~')
                    inwater = True 

                if titleType == 'T':
                    if inwater == True:
                        raft = False
                        inwater = False
                    else:
                        raft = True
                    myMap.setType(titlepos, ' ')
                if titleType == '*':
                    dynamite -= 1
                    myMap.setType(titlepos, ' ')
            else:
                myMap1 = map.copy()
                myMap = map

                
            if (titlepos, key, axe, dynamite, raft, inwater) in went:
                continue

            went.add((titlepos, key, axe, dynamite, raft, inwater))
            if (map.getType(Map.north(originPos)) == '~' or map.getType(Map.south(originPos)) == '~'or map.getType(Map.west(originPos)) == '~' or map.getType(Map.east(originPos)) == '~')and (self.explore_again == True):
                self.explore_again = False
                self.swimming = True
                self.map.transferground()
                return []
            distancetopos = [Map.north(titlepos), Map.south(titlepos), Map.east(titlepos), Map.west(titlepos)]
            distancetopos.sort(key=lambda x:Map.distance(targetPos, x),reverse=True)
            for pos in distancetopos:
                possiblemove.append((path+[pos], key, axe, dynamite,raft, inwater,myMap))
        return []
## produce the path to step to send the commond
    def processPath(self, originPos, targetPos, key, axe, dynamite, raft):
        
        map = self.map
        
        possiblemove = [([originPos], key, axe, dynamite, raft, map)]
        went = set()
        while(possiblemove):
            path, key, axe, dynamite, raft, map = possiblemove[0]
            possiblemove = possiblemove[1:]
            titlepos = path[-1]
            if titlepos == targetPos:
                return path

            canwalk = {' ', 'k', 'a', 'd'}
            if key:
                canwalk.add('-')
            if axe:
                canwalk.add('T')

            titleType = map.getType(titlepos)

            if titleType not in canwalk:
                continue
            
            if titleType in 'kda~':
                myMap = map.copy()
                if titleType == 'k':
                    key = True
                    myMap.setType(titlepos, ' ')
                if titleType == 'd':
                    dynamite += 1
                    myMap.setType(titlepos, ' ')
                if titleType == 'a':
                    axe = True
                    myMap.setType(titlepos, ' ')
            else:
                myMap = map
            
            if (titlepos, key, axe, dynamite, raft, myMap) in went:
                continue
            went.add((titlepos, key, axe, dynamite, raft, myMap))
            
            distancetopos = [Map.north(titlepos), Map.south(titlepos), Map.east(titlepos), Map.west(titlepos)]
            distancetopos.sort(key=lambda x:Map.distance(targetPos, x),reverse=True)
            for pos in distancetopos:
                possiblemove.append((path+[pos], key, axe, dynamite,raft, myMap))
            
            
        return []
##find the return path to (0,0)

    def findreturnPath(self, map, titlepos,key, axe, dynamite, raft):
        inwater1 = False
        raft = False
        myMap1= map.copy()
        possiblemove1 = [([titlepos], key, axe, dynamite, raft,inwater1, myMap1)]
        went1 = set()
        while(possiblemove1):
            
            path1, key1, axe1, dynamite1, raft1,inwater1, myMap1 = possiblemove1[0]
            print(path1)
            possiblemove1 = possiblemove1[1:]
            titlepos1 = path1[-1]
            
            canwalk1 = {' ', 'k', 'a', 'd','$'}
            if key1:
                canwalk1.add('-')
            if axe1:
                canwalk1.add('T')
            if dynamite1:
                canwalk1.add('*')
            if raft1:
                canwalk1.add('~')

            titleType1 = map.getType(titlepos1)
            if titlepos1 == (0,0):
                return path1

            if titleType1 not in canwalk1:
                continue

                
            if (titleType1 == ' ') and (inwater1 == True):
                raft1 = False
                inwater1 = False               
                
            if titleType1 in 'kda~T*':

                if titleType1 == 'k':
                    key1 = True
                    myMap1.setType(titlepos1, ' ')
                if titleType1 == 'd':
                    dynamite1 += 1
                    myMap1.setType(titlepos1, ' ')
                if titleType1 == 'a':
                    axe1 = True
                    myMap1.setType(titlepos1, ' ')

                if titleType1 == '~':
                    myMap1.setType(titlepos1,'~')
                    inwater1 = True 

                if titleType1 == 'T':
                    
                    raft1 = True
                    myMap1.setType(titlepos1, ' ')
                if titleType1 == '*':
                    dynamite1 -= 1
                    myMap1.setType(titlepos1, ' ')


                
            if (titlepos1, key1, axe1, dynamite1, raft1, inwater1) in went1:
                continue

            went1.add((titlepos1, key1, axe1, dynamite1, raft1, inwater1))
            distancetopos1 = [Map.north(titlepos1), Map.south(titlepos1), Map.east(titlepos1), Map.west(titlepos1)]
            distancetopos1.sort(key=lambda x:Map.distance((0,0), x),reverse=True)
            for pos1 in distancetopos1:
                possiblemove1.append((path1+[pos1], key1, axe1, dynamite1,raft1, inwater1,myMap1))
        return []
## exclude the wrong wall that make the agent can not get back to original position        
    def removeWrongDynamite(self, map, key, axe, dynamite, raft):
        myMap = map.copy()
        resource = myMap.resourcePos()
        if map.hastreasure():
            resource.append(map.treasurePos())
        dynamite += self.map.dynamitesOnGround()
        originPos = map.myposition()
        canwalk = {'$', 'k', 'a', ' ', 'd'}
        if key:
            canwalk.add('-')
        if axe:
            canwalk.add('T')
        if raft:
            canwalk.add('~')
        area = self.walkablearea(originPos, myMap, canwalk, dynamite, raft)
        canwalk.add('*')
        s = set()
        for resPos in resource:
            queue = [(resPos, dynamite)]
            walked = set()
            
            while(queue):
                
                pos , tmpdynamite = queue[0]
                queue = queue[1:]
                if pos in walked:
                    
                    continue

                if myMap[pos] not in canwalk:
                    continue
                walked.add(pos)
                if myMap[pos] == '*':
                    s.add(pos)
                    if tmpdynamite == 1:
                        continue
                    else:
                        tmpdynamite -= 1
                if pos in area:
                    break
                queue.append((Map.north(pos), tmpdynamite))
                queue.append((Map.east(pos), tmpdynamite))
                queue.append((Map.south(pos), tmpdynamite))
                queue.append((Map.west(pos), tmpdynamite))

        for pos in myMap:
            if myMap[pos] == '*':
                if  pos not in s:
                    myMap.setType(pos, 'x')
##                if self.inthewater == True:
##                    if self.map.getType(Map.north(pos)) == '~' or self.map.getType(Map.south(pos)) == '~' or self.map.getType(Map.west(pos)) == '~' or self.map.getType(Map.east(pos)) == '~':
##                        myMap.setType(pos, 'x')
        return myMap

   
    def walkablearea(self, originPos, map, canwalk, dynamite, raft):
        possiblemove = [(originPos)]
        area = set()
        while(possiblemove):
            pos = possiblemove.pop()
            if map[pos] not in canwalk:
                continue
            if pos in area:
                continue

            area.add(pos)
            possiblemove.append((Map.north(pos)))
            possiblemove.append((Map.east(pos)))
            possiblemove.append((Map.south(pos)))
            possiblemove.append((Map.west(pos)))
        return area
##explore around to get more information for map
    def explore(self):
        bordertitles = self.boundary()
        if bordertitles:
            borderPathes = self.findPaths(bordertitles)[0]
            self.processStep([x for x in borderPathes])
        return len(bordertitles) != 0
    def exploreagain(self):
        if self.swimming == True:
            return
        bordertitles = self.boundary2()
        if bordertitles:
            borderPathes = self.findPaths(bordertitles)[0]
            self.processStep2([x for x in borderPathes])
        return len(bordertitles) != 0
    def explorefinal(self):
        bordertitles = self.unexploreinMap()
        if bordertitles:
            borderPathes = self.findPaths(bordertitles)[0]
            self.map.setType(self.map.pos, '~')
            self.processStep2([x for x in borderPathes])
        return len(bordertitles) != 0
    def exploreonwater(self):

        bordertitles = self.boundaryonwater()
        if bordertitles:
            borderPathes = self.findPaths(bordertitles)[0]
            self.processStep3([x for x in borderPathes])
        return len(bordertitles) != 0
##find the border position without using raft
    def boundary(self):
        pos = self.map.myposition()

        canwalk = {' ', 'k', 'a', 'd'}
        if self.hasKey:
            canwalk.add('-')
        if self.hasAxe:
            canwalk.add('T')

        walked = set()
        possiblemove = [pos]

        bordertitle = []
        while(possiblemove):
            pos = possiblemove.pop()

            if pos in walked:
                continue
            walked.add(pos)
            if self.map[pos] not in canwalk:
                continue
            if 'W' in self.map.surroundingType(pos):
                bordertitle.append(pos)
            possiblemove.append(Map.north(pos))
            possiblemove.append(Map.west(pos))
            possiblemove.append(Map.south(pos))
            possiblemove.append(Map.east(pos))
        
        return bordertitle
##find the border position can use raft
    def boundary2(self):

        pos = self.map.myposition()

        canwalk = {' ', 'k', 'a', 'd'}
        if self.hasKey:
            canwalk.add('-')
        if self.hasAxe:
            canwalk.add('T')
        if self.hasraft:
            canwalk.add('~')
        if self.inthewater:
            canwalk.add('~')

        walked = set()
        possiblemove = [pos]

        bordertitle = []
        while(possiblemove):
            pos = possiblemove.pop()

            if pos in walked:
                continue
            walked.add(pos)
            if self.map[pos] not in canwalk:
                continue
            if 'W' in self.map.surroundingType(pos):
                bordertitle.append(pos)
            possiblemove.append(Map.north(pos))
            possiblemove.append(Map.west(pos))
            possiblemove.append(Map.south(pos))
            possiblemove.append(Map.east(pos))
        return bordertitle
    def boundaryonwater(self):
        pos = self.map.myposition()

        canwalk = {' ', '~'}

        walked = set()
        possiblemove = [pos]

        bordertitle = []

        while(possiblemove):
            pos = possiblemove.pop()
            if pos in walked:
                continue
            walked.add(pos)
            if self.map[pos] not in canwalk:
                
                continue
            if 'W' in self.map.surroundingType(pos):
                bordertitle.append(pos)
            possiblemove.append(Map.north(pos))
            possiblemove.append(Map.west(pos))
            possiblemove.append(Map.south(pos))
            possiblemove.append(Map.east(pos))
        
        return bordertitle
##find the area have not explored in the map
    def unexploreinMap(self):
        pos = self.map.myposition()
        walked = set()
        possiblemove = [pos]
        bordertitle = []
        canwalk = {' ', 'k', 'a', 'd','T','~', '*'}
        while(possiblemove):
            pos = possiblemove.pop()
            if pos in walked:
                continue
            if self.map[pos] not in canwalk:
                continue
            walked.add(pos)
            if ('W' == self.map.getType(Map.north(pos)) )or( 'W' == self.map.getType(Map.south(pos)) )or ('W' == self.map.getType(Map.west(pos)) )or ('W' == self.map.getType(Map.east(pos))):
                if self.map.getType(pos) != 'W' and self.map.getType(pos) != '.':
                    bordertitle.append(pos)
            possiblemove.append(Map.north(pos))
            possiblemove.append(Map.west(pos))
            possiblemove.append(Map.south(pos))
            possiblemove.append(Map.east(pos))
        
        return bordertitle
##connect the boundary in a path        
    def findPaths(self, titles):
        pathes = [[x] for x in titles]
        while(True):
            flag = False
            for i in range(len(pathes)):
                for j in range(len(pathes)):
                    if i == j:
                        continue
                    elif Map.distance(pathes[i][-1], pathes[j][0]) == 1:
                        pathes[i] += pathes[j]
                        pathes = pathes[:j] + pathes[j+1:]
                        flag = True
                        break
                    elif Map.distance(pathes[j][-1], pathes[i][0]) == 1:
                        pathes[j] += pathes[i]
                        pathes = pathes[:i] + pathes[i+1:]
                        flag = True
                        break
                if flag:
                    break
            if flag:
                continue
            break
        return [path for path in pathes if len(path) == max([len(path) for path in pathes])]
##produce a path to step function to send command    
    def processStep(self, path):
        head = path[0]
        if head != self.map.myposition():
            walkPath = self.processPath(self.map.myposition(), head, self.hasKey, self.hasAxe, 0, False)
            if walkPath:
                path = walkPath[:-1] + path
        if path[0] == self.map.myposition():
            for title in path[1:]:
                self.step(title)
    def processStep2(self, path):
        head = path[0]
        if head != self.map.myposition():
            walkPath = self.findsolution(self.map.myposition(), head, self.hasKey, self.hasAxe, 0, self.hasraft)
            if walkPath:
                path = walkPath[:-1] + path
        if path[0] == self.map.myposition():
            for title in path[1:]:
                self.step(title)
    def processStep3(self, path):
        
        head = path[0]
        if head != self.map.myposition():
            walkPath = self.processPathonwater(self.map.myposition(), head, self.hasKey, self.hasAxe, 0, self.hasraft)
            if walkPath:
                path = walkPath[:-1] + path
        if path[0] == self.map.myposition():
            for title in path[1:]:               
                self.step(title)
##send the command
    def step(self, targetPos):
        command = ''
        myposition = self.map.myposition()
        
        if Map.distance(myposition,targetPos) != 1:
            raise Exception

        turn = Map.directionDiff(myposition, targetPos, self.map.getFacing())

        if abs(turn) == 2:
            command += 'rr'
        elif turn == 1 or turn == -3:
            command += 'r'
        elif turn == 3 or turn == -1:
            command += 'l'

        targetType = self.map[targetPos]
        if targetType == 'T':
            command += 'c'
            self.hasraft = True
            
        elif targetType == '-':
            command += 'u'
        elif targetType == 'd':
            self.hasdynamite += 1
        elif targetType == 'a':
            self.hasAxe = True
        elif targetType == 'k':
            self.hasKey = True
        elif targetType == '*':
            command += 'b'
            self.hasdynamite -= 1
        elif targetType == '~':
            self.inthewater = True
        elif targetType == ' ' and self.inthewater == True:
            self.hasraft = False
            self.inthewater = False
        elif targetType == 'T' and self.inthewater == True:
            self.hasraft = False
            self.inthewater = False
        self.map.setType(targetPos, ' ')

        command += 'f'


        for c in command:
            if c == 'r':
                self.map.setFacing(self.map.getFacing()+1)
            elif c == 'l':
                self.map.setFacing(self.map.getFacing()-1)
            elif c == 'f':
                self.map.setPos(targetPos)
            self.pipe.send(c)
            if self.swimming == True:
                self.map.transferground()
            
            print(self.map.printMap())

class Pipe:
    def __init__(self, port, map):
        self.port = port
        self.conn = None
        self.map = map
        self.connection()

    def connection(self):
        self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn.connect(('127.0.0.1', self.port))
        self.receiveonline()

    def send(self, message):
        m = str.encode(message)
        self.conn.send(m)
        self.receiveonline()

    def receiveonline(self):
        d = b''
        while(True):
            d += self.conn.recv(1024)
            if len(d) == 24:
                nodelist = list(d.decode('utf-8'))
                receivedata = nodelist[:12] + ['^'] + nodelist[12:]
                grid = [receivedata[5*i:5*i+5] for i in range(5)]
                self.map.mapScope(grid)
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=int, help='port number of server')
    args = parser.parse_args()
    a = Agent(args.p)
    a.begin()

