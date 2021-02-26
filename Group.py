import Box
import numpy as np


class Group:
    """ Class handling group made of multiple persons
    """
    def __init__(self):
        self.members = []
        self.containgBox = Box.setXYWH([-1,-1,-1,-1])

    def addPerson(self, person):
        self.members.append(person)

        if(len(self.members)==1):
            self.containgBox = Box.setXYWH(person.box.getXYWH())
        else:
            newStart = tuple(np.minimum(self.containgBox.startPoint(),person.box.startPoint()))
            newEnd = tuple(np.maximum(self.containgBox.endPoint(),person.box.endPoint()))

            self.containgBox = Box.setStartEnd(newStart,newEnd)

    def getContaingBox(self):
        return  self.containgBox 

    def personCount(self):
        return len(self.members)

def belongToGroup(person, group):
    inTheGroup = False

    length = len(group.members)
    i = 0

    while(i<length and not inTheGroup):
        heightDifference  = group.members[i].box.height / person.box.height

        if person.box.doOverlap(group.members[i].box) and (0.8 <= heightDifference <= 1.2):
            inTheGroup = True
        i = i+1

    return inTheGroup


def isSubGroup(innerGroup, outerGroup):
    inTheGroup = False
    innerGroupBox = innerGroup.getContaingBox()

    length = len(outerGroup.members)
    i = 0

    while(i<length and not inTheGroup):
        heightDifference  = outerGroup.members[i].box.height / innerGroupBox.height

        if innerGroupBox.doOverlap(outerGroup.members[i].box) and (0.8 <= heightDifference <= 1.2):
            inTheGroup = True
        i = i+1

    return inTheGroup

def mergeGroups(groupA, groupB):
    result = Group()
    
    for p in  groupA.members:
        result.addPerson(p)

    for p in  groupB.members:
        result.addPerson(p)
    
    return result


    
