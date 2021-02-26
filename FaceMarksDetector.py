""" Libs """
import tensorflow as tf
import numpy as np
import cv2

""" Custom files """
import FaceDetector

# debug lvl
# 0 - no debug
# 1 - informational
# 2 - detailed log
# 3 - verbose 

debug = 0

class MarkDetector:
    """Facial landmark detector by Convolutional Neural Network"""

    def __init__(self, saved_model, dnn_proto_text, dnn_model):
        """Initialization saved_model directory containg the model for the makr detector
            dnn_proto_text configuration file for the face detector
            dnn_model caffe model file fo the face detector
        """
        # saved_model='models/pose_model', 
        # dnn_proto_text='models/deploy.prototxt',
        # dnn_model='models/res10_300x300_ssd_iter_140000.caffemodel'):
        # A face detector is required for mark detection.
        self.face_detector = FaceDetector.FaceDetector(dnn_proto_text, dnn_model)

        self.cnn_input_size = 128
        self.marks = None

        # Restore model from the saved_model file.
        #self.model = keras.models.load_model(saved_model)
        self.model = tf.keras.models.load_model(saved_model, compile=False)

    @staticmethod
    def draw_box(image, boxes, box_color=(255, 255, 255)):
        """Draw square boxes on image"""
        for box in boxes:
            cv2.rectangle(image,
                          (box[0], box[1]),
                          (box[2], box[3]), box_color, 3)

    @staticmethod
    def move_box(box, offset):
        """Move the box to direction specified by vector offset"""
        left_x = box[0] + offset[0]
        top_y = box[1] + offset[1]
        right_x = box[2] + offset[0]
        bottom_y = box[3] + offset[1]
        return [left_x, top_y, right_x, bottom_y]

    @staticmethod
    def get_square_box(box):
        """Get a square box out of the given box, by expanding it."""
        left_x = box[0]
        top_y = box[1]
        right_x = box[2]
        bottom_y = box[3]

        box_width = right_x - left_x
        box_height = bottom_y - top_y

        # Check if box is already a square. If not, make it a square.
        diff = box_height - box_width
        delta = int(abs(diff) / 2)

        if diff == 0:                   # Already a square.
            return box
        elif diff > 0:                  # Height > width, a slim box.
            left_x -= delta
            right_x += delta
            if diff % 2 == 1:
                right_x += 1
        else:                           # Width > height, a short box.
            top_y -= delta
            bottom_y += delta
            if diff % 2 == 1:
                bottom_y += 1

        # Make sure box is always square.
        assert ((right_x - left_x) == (bottom_y - top_y)), 'Box is not square.'

        return [left_x, top_y, right_x, bottom_y]

    @staticmethod
    def box_in_image(box, image):
        """Check if the box is in image"""
        rows = image.shape[0]
        cols = image.shape[1]
        return box[0] >= 0 and box[1] >= 0 and box[2] <= cols and box[3] <= rows

    def extract_cnn_facebox(self, image,threshold):
        """Extract face area from image."""
        _, raw_boxes = self.face_detector.get_faceboxes(image, threshold)

        a = []
        for box in raw_boxes:
            # Move box down by 10% of height.
            offset_y = int(abs((box[3] - box[1]) * 0.1))
            box_moved = self.move_box(box, [0, offset_y])

            # Make box square.
            facebox = self.get_square_box(box_moved)

            if self.box_in_image(facebox, image):
                a.append(facebox)

        return a
      
    def detect_marks(self, image_np):
        """Detect marks from image"""

        # # Actual detection.
        predictions = self.model.signatures["predict"](tf.constant(image_np, dtype=tf.uint8))

        # Convert predictions to landmarks.
        marks = np.array(predictions['output']).flatten()[:136]
        marks = np.reshape(marks, (-1, 2))

        return marks

    @staticmethod
    def draw_marks(image, marks, color=(255, 255, 255)):
        """Draw mark points on image"""
        for mark in marks:
            cv2.circle(image, (int(mark[0]), int(mark[1])), 1, color, -1, cv2.LINE_AA)
            
def computeLineOfSigth(img, rotation_vector, translation_vector, camera_matrix, color=(255, 255, 0), line_width=2):
    point_3d = []
    dist_coeffs = np.zeros((4,1))
    rear_size = img.shape[1] // 4
    rear_depth = rear_size
    #NOTE left and rigth are from the camera pov
    #0 bottom left
    point_3d.append((-rear_size, -rear_size, rear_depth))
    #1 top left
    point_3d.append((-rear_size, rear_size, rear_depth))
    #2 top rigth
    point_3d.append((rear_size, rear_size, rear_depth))
    #3 bottom rigth
    point_3d.append((rear_size, -rear_size, rear_depth))
    #4 bottom left
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = img.shape[1]
    front_depth = front_size*2
    #5 bottom left
    point_3d.append((-front_size, -front_size, front_depth))
    #6 top left
    point_3d.append((-front_size, front_size, front_depth))
    #7 top rigth
    point_3d.append((front_size, front_size, front_depth))
    #8 bottom rigth
    point_3d.append((front_size, -front_size, front_depth))
    #9 bottom left
    point_3d.append((-front_size, -front_size, front_depth))

    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

    # Map to 2d img points
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      dist_coeffs)
    point_2d = np.int32(point_2d.reshape(-1, 2))

    if debug > 1 :
        #  Draw the 2 squares
        cv2.polylines(img, [point_2d], True, color, line_width, cv2.LINE_AA)

        # drawing the diagonals
        # inner square
        cv2.line(img, tuple(point_2d[0]), tuple(point_2d[2]), color, line_width, cv2.LINE_AA)    
        cv2.line(img, tuple(point_2d[1]), tuple(point_2d[3]), color, line_width, cv2.LINE_AA)    

        #outer square
        cv2.line(img, tuple(point_2d[5]), tuple(point_2d[7]), color, line_width, cv2.LINE_AA)    
        cv2.line(img, tuple(point_2d[6]), tuple(point_2d[8]), color, line_width, cv2.LINE_AA)    
        
        # draw the lines connecting the edges fo the 2 squares
        cv2.line(img, tuple(point_2d[1]), tuple(point_2d[6]), color, line_width, cv2.LINE_AA)    
        cv2.line(img, tuple(point_2d[2]), tuple(point_2d[7]), color, line_width, cv2.LINE_AA)
        cv2.line(img, tuple(point_2d[3]), tuple(point_2d[8]), color, line_width, cv2.LINE_AA)

    

    # center of the outer diagonals
    center_out = (point_2d[8][0] +  point_2d[9][0])//2 , (point_2d[8][1] +  point_2d[7][1])//2
    # center of the inner diagonals
    center_in = (point_2d[3][0] +  point_2d[4][0])//2 , (point_2d[3][1] +  point_2d[2][1])//2

    
    return (center_in, center_out)
