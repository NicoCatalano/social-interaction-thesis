#https://towardsdatascience.com/real-time-head-pose-estimation-in-python-e52db1bc606a
""" Libs """
import cv2
import numpy as np
import tensorflow as tf
import sys
import os.path

"""Custom files"""
import FaceMarksDetector
import FaceDetector
import Box


def isValidVideoStream(source):
    webcamID = int(source)
    capWebcam = cv2.VideoCapture(webcamID) 
    cap = cv2.VideoCapture(source) 

    if cap is not None and cap.isOpened() :
        return (True, cap)
    elif capWebcam is not None and capWebcam.isOpened() :
        return (True, capWebcam)
    return (False, None)

class GazeDetector:
    def __init__(self, imageShape, debug = 0):
        # debug lvl
        # 0 - no debug
        # 1 - informational
        # 2 - detailed log
        # 3 - verbose 

        self.debug = debug

        modelFile  = "models/res10_300x300_ssd_iter_140000.caffemodel"
        configFile = "models/deploy.prototxt.txt"
        saved_model= "models/pose_model" #mark detector model dir

        if not (os.path.isfile(configFile)):
            print("Config file for face detector does not exist")
            exit()
        if not (os.path.isfile(modelFile)):
            print("Caffe mode file for face detector does not exist")
            exit()

        self.mark_detector = FaceMarksDetector.MarkDetector(saved_model,configFile,modelFile)

        # 3D model points
        self.model_points = np.array([
                                    (0.0, 0.0, 0.0),             # Nose tip
                                    (0.0, -330.0, -65.0),        # Chin
                                    (-225.0, 170.0, -135.0),     # Left eye left corner
                                    (225.0, 170.0, -135.0),      # Right eye right corne
                                    (-150.0, -150.0, -125.0),    # Left Mouth corner
                                    (150.0, -150.0, -125.0)      # Right mouth corner
                                ])

        # Camera internals
        focal_length = imageShape[1]
        center = (imageShape[1]/2, imageShape[0]/2)
        self.camera_matrix = np.array(
                                [[focal_length, 0,       center[0]],
                                [0,        focal_length, center[1]],
                                [0,            0,           1]], 
                                dtype = "double"
                                )
    
    def get_faceboxes(self,img,threshold):
        """ Gives a list of bounding boxes containg faces in the given image"""
        faces = self.mark_detector.extract_cnn_facebox(img,threshold)
        


        faceboxes = []
        for face in faces:
            start = (face[0],face[1])
            end = (face[2],face[3])
            box =  Box.setStartEnd(start,end)

            faceboxes.append(box)
        
        return faceboxes
    
    def getGazeDirection(self,img, facebox):
        """ Returns marks, (x1,x2) x1 and x2 are points identifying a line ortogonal to the face plane"""
        facebox_list = facebox.getList()
        
        #extrat face box and downsampling
        face_img = img[facebox_list[1]: facebox_list[3],facebox_list[0]: facebox_list[2]]
        face_img = cv2.resize(face_img, (128, 128))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        #marks detection
        marks = self.mark_detector.detect_marks([face_img])
        
        #scale and move back marks in original image coordinate
        marks *= (facebox_list[2] - facebox_list[0])
        marks[:, 0] += facebox_list[0]
        marks[:, 1] += facebox_list[1]
        shape = marks.astype(np.uint)

        #TODO:consider different points for surgery masks
        image_points = np.array([
                                shape[30],     # Nose tip
                                shape[8],      # Chin
                                shape[36],     # Left eye left corner
                                shape[45],     # Right eye right corne
                                shape[48],     # Left Mouth corner
                                shape[54]      # Right mouth corner
                            ], dtype="double")
        
        if self.debug > 1:
            for p in image_points:
                cv2.circle(img, (int(p[0]), int(p[1])), 3, (0,0,255), -1)

        #Solving PnP
        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(self.model_points, image_points, self.camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP)
        
        #Get a ortogal to tha face plane - x1 and x2 are two points definig the line in the projected space
        #TODO: remove and make a line going out from eyes
        x1, x2 = FaceMarksDetector.computeLineOfSigth(img, rotation_vector, translation_vector, self.camera_matrix)

        if self.debug > 0:
            #display the line
            cv2.line(img, tuple(x1), tuple(x2), (255, 255, 0), 2)

            for (x, y) in shape:
                cv2.circle(img, (x, y), 4, (255, 255, 0), -1)

        if self.debug > 0:
            self.mark_detector.draw_marks(img, marks, color=(0, 255, 0))
            
        return marks, (x1, x2)


# Main for testing pourpose

""" TESTING images """
if __name__ == "__main__":
    threshold = 0.5
    inputFileName = str(sys.argv[1])
    img = cv2.imread(inputFileName)

    gd = GazeDetector(img.shape,3)
    
    faceboxes = gd.get_faceboxes(img, threshold)
    print("Face detected:",len(faceboxes))
    for facebox in faceboxes:
        x1, x2 = gd.getGazeDirection(img,facebox)

    cv2.imshow('img', img)
    
    # Display output
    cv2.waitKey(0)
    cv2.destroyAllWindows()

""" TESTING video """
"""
if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')


    if len(sys.argv) != 2:
        print("usage:",sys.argv[0],"videoStream (filePath or webcamID)")
        exit()

    streamFilename = sys.argv[1]
    validVideo, cap = isValidVideoStream(streamFilename)

    if not (validVideo):
        print(streamFilename, "is not a valid video stream, please provide a video file or a webcam ID")
        exit()

    ret, img = cap.read()
    gd = GazeDetector(img.shape,1)

    # process each frame
    while True:
        ret, img = cap.read()
        if ret == True:
            faceboxes = gd.get_faceboxes(img, threshold)
            for facebox in faceboxes:
                x1, x2 = gd.getGazeDirection(img,facebox)

            cv2.imshow('img', img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            print("can't open video stream")
            break

    cv2.destroyAllWindows()
    cap.release()

"""
