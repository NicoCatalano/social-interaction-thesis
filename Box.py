class Box:
    def __init__(self, center_x, center_y, width, height):
        self.center_x = center_x
        self.center_y = center_y
        self.width = width
        self.height = height

    def startPoint(self):
        x = round(self.center_x - self.width / 2)
        y = round(self.center_y - self.height / 2)

        return (x,y)

    def endPoint(self):
        x = round(self.center_x + self.width / 2)
        y = round(self.center_y + self.height / 2)

        return (x,y)

    def getXYWH(self):
        x = round(self.center_x - self.width / 2)
        y = round(self.center_y - self.height / 2)

        return [x, y, self.width, self.height]
    
    def getROI(self, img):
        x1 = self.startPoint()[0]
        y1 = self.startPoint()[1]
        x2 = self.endPoint()[0]
        y2 = self.endPoint()[1]

        return img[y1:y2, x1:x2]

    def getROIcoordinates(self):
        """if (x1,y1) and (x2,y2) are the two opposite vertices the ROI coordinates are [y1,y2, x1,x2]"""
        x1 = self.startPoint()[0]
        y1 = self.startPoint()[1]

        x2 = self.endPoint()[0]
        y2 = self.endPoint()[1]

        return [y1,y2, x1,x2]

    def contains(self, secondBox):
        """return true if the second box is entirly in the box """
        #first box (outer)
        outerX1 = self.startPoint()[0]
        outerY1 = self.startPoint()[1]

        outerX2 = self.endPoint()[0]
        outerY2 = self.endPoint()[1]
        #second box (inner)
        innerX1 = secondBox.startPoint()[0]
        innerY1 = secondBox.startPoint()[1]

        innerX2 = secondBox.endPoint()[0]
        innerY2 = secondBox.endPoint()[1]
        
        if outerX1 <= innerX1 and outerY1 <= innerY1 and outerX2 >= innerX2 and outerY2 >= innerY2 :
            return True
        
        return False
   
    def doOverlap(self, secondBox): 
        # If one rectangle is on left side of other 
        
        if(self.startPoint()[0] > secondBox.endPoint()[0] or secondBox.startPoint()[0] > self.endPoint()[0]): 
            return False
    
        # If one rectangle is above other 
        if(self.startPoint()[1] > secondBox.endPoint()[1] or secondBox.startPoint()[1] > self.endPoint()[1]): 
            return False
    
        return True
    def getList(self):
        """ return a list  of the coordinate of the 2 extreme points """
        start = self.startPoint()
        end = self.endPoint()
        return [start[0],start[1],end[0],end[1]]



def setXYWH(xywh):
    return Box(round(xywh[0]+xywh[2]//2), round(xywh[1]+xywh[3]//2), xywh[2], xywh[3])

def setStartEnd(start, end):
    x_start = start[0]
    y_start = start[1]

    x_end = end[0]
    y_end = end[1]

    width = x_end - x_start
    height = y_end - y_start
    
    x_center = round(x_start + (width) / 2)
    y_center = round(y_start + (height) / 2)

    return Box(x_center, y_center, width, height)

