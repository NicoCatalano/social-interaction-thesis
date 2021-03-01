""" libs """
import cv2
import numpy as np
import pandas as pd
import sys

""" Custom Files """
import Box
import Group
import Person
import GazeDetector

# debug lvl
# 0 - no debug
# 1 - informational
# 2 - detailed log
# 3 - verbose 

debug = 0

def detectPeopleGroups(net, gd, image, faceTrashold):
    Width = image.shape[1]
    Height = image.shape[0]

    # create input blob 
    # set input blob for the network
    net.setInput(cv2.dnn.blobFromImage(image, 0.00392, (416,416), (0,0,0), True, crop=False))

    # run inference through the network
    # and gather predictions from output layers
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    outs = net.forward(output_layers)

    # confidences is used to keep track f the confidence of each detection, and used for nn maximum supporession
    confidences = []
    boxes = []

    #create bounding box 
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)

            #if person
            if class_id == 0:
            
                confidence = scores[class_id]
                if confidence > 0.1:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    
                    box = Box.Box(center_x,center_y,w,h)

                    confidences.append(float(confidence))
                    boxes.append(box.getXYWH())

    #non maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.1)


    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faceboxes = gd.get_faceboxes(image,faceTrashold)

    groups = []
    #displaying bboxes
    for i in indices:
        i = i[0]
        box = Box.setXYWH(boxes[i])
        
        if debug > 1:
            print("found person in", box.getXYWH())

        p = Person.Person(i,box)


        if debug > 1:
            print("Face found:", len(faceboxes))

        #assigning each face to the corresponding person    
        for facebox in faceboxes:
            if p.box.contains(facebox) :
                marks, lineOfSigth, euler_angles = gd.getGazeDirection(image,facebox)
                p.markpoints = marks
                p.lineOfSigth = lineOfSigth
                p.euler_angles = euler_angles
                
                if debug > 2 :
                    print("Face detected in ",facebox)
                #the assigned face can't belong to another person
                faceboxes.remove(facebox)
                #each person can only have one face
                break


        #groups of people
        if debug > 2 :
            print("Adding person id", p.id)

        """
            If there are no grops I create a new group and add to it the new found person
            If there are already some gropus I will go trough each one and check if the newly found person is part of that group.

            If the new founded person isn't belonging to any group I create a new one and add the person to it        
        """
        if(len(groups) == 0):
            if debug > 2 :
                print("Groups list empy, adding new group (1 person)",p.box.startPoint(), p.box.endPoint())
            g = Group.Group()
            g.addPerson(p)
            groups.append(g)
        else:
            groupsNumber = len(groups)
            group_index = 0
            added = False
            
            if debug > 2 :
                print("found ", groupsNumber, " groups")
            
            while(group_index<groupsNumber and not added): 
                g = groups[group_index]
                
                if Group.belongToGroup(p,g):
                    if debug > 1 :
                        print("adding to existing group",group_index)
                    g.addPerson(p)
                    added = True

                group_index += 1

            if(group_index == groupsNumber and not added):
                g = Group.Group()
                g.addPerson(p)
                groups.append(g)
                
                if debug > 1 :
                    print("adding new group")
      


    #merging gruops
    group_index = 0
    while(group_index<len(groups)):
        group_index_toCheck = group_index + 1
        box = groups[group_index].getContaingBox()

        while(group_index_toCheck<len(groups)):
            if Group.isSubGroup(groups[group_index], groups[group_index_toCheck]):
                mergedGroup = Group.mergeGroups(groups[group_index],groups[group_index_toCheck])
                groups[group_index] = mergedGroup 
                groups.pop(group_index_toCheck)
            else:
                group_index_toCheck += 1

        group_index += 1


    return groups

def displayGroups(image, groups):
    #displaying groups' boxes
    group_index = 0
    while(group_index<len(groups)):
        box = groups[group_index].getContaingBox()

        cv2.rectangle(image, box.startPoint(), box.endPoint(), (0, 255, 0), 2)
        cv2.putText(image,  str(group_index), box.startPoint(), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        people = groups[group_index].members
        for person in people:
            if person.hasFace():
                start,end = person.lineOfSigth
                cv2.line(image, start , end , (255,0,0), 2, cv2.LINE_AA) 

                markpoints = person.markpoints
                for m in markpoints:
                    cv2.circle(image, tuple(m), 1, (0,255,255))

        group_index += 1

    return image 