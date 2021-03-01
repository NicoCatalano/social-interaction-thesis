""" Custom Files """
import Box

class Person:
    """ Person descriptor, contains bounding box ifnormation and gaze direction"""
    def __init__(self,id, box):
        self.box = box
        self.id = id
        #tuple of points defining a line ortogonal to the face - line of sigth (start point and end point) 
        self.lineOfSigth = None
        self.markpoints = None
        self.euler_angles = None
    
    def hasFace(self):
        if self.lineOfSigth is None:
            return False
        return True
    
