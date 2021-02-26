""" Libs """
import sys
import cv2
import matplotlib.pyplot as plt
import tqdm

""" Custom files """
import Detector
import GazeDetector 


def detectInEachFrame(net, video, maxFrames,faceTrashold):
    """ 
        Runs the predictor on every frame in the video,
        and returns the frame with the predictions drawn.
    """
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    shape = (width,height)

    gd = GazeDetector.GazeDetector(shape)

    readFrames = 0
    while True:
        hasFrame, frame = video.read()
        if not hasFrame:
            break

        # Make sure the frame is colored --- check
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # run the detector
        groups = Detector.detectPeopleGroups(net,gd,frame,faceTrashold)
        
        # overlap boxes on the image
        visualization = Detector.displayGroups(frame, groups)
        
        # Convert Matplotlib RGB format to OpenCV BGR format --- check
        visualization = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)

        yield visualization

        readFrames += 1
        if readFrames > maxFrames:
            break


def processVideo(net, inName, faceTrashold, outName = 'out.mp4'):
    """
    It call the detector and produce the video output 
    """
    # Extract video properties
    video = cv2.VideoCapture(inName)

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))


    # Initialize video writer
    video_writer = cv2.VideoWriter(outName, fourcc=cv2.VideoWriter_fourcc(*"mp4v"), fps=float(frames_per_second), frameSize=(width, height), isColor=True)

    # Enumerate the frames of the video
    for frame in tqdm.tqdm(detectInEachFrame(net, video, num_frames,faceTrashold), total=num_frames):

        # Write to video file
        video_writer.write(frame)

    # Release resources
    video_writer.release()


""" TEST on single image """
def testOnImage():
    inputFileName = str(sys.argv[1])
    img = cv2.imread(inputFileName)


    #load network configuration, weights and object names
    classes = None
    with open('coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    net = cv2.dnn.readNet('darknet/yolov3.weights', 'darknet/cfg/yolov3.cfg')

    shape = (img.shape[0], img.shape[1])

    #initilize gaze dtector
    gd = GazeDetector.GazeDetector(shape,3)

    # Make sure the frame is colored --- check
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # run the detector
    groups = Detector.detectPeopleGroups(net,gd,img,faceTrashold = 0.01)

    # overlap boxes on the image
    visualization = Detector.displayGroups(img, groups)

    # Convert Matplotlib RGB format to OpenCV BGR format --- check
    visualization = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)

    # Display output
    cv2.imshow('out',visualization)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #saving out
    cv2.imwrite("out_person_detector.png", visualization)



""" TEST on video """
def testOnVideo():
    argumentsCount = len(sys.argv) - 1
    faceTrashold = 0.15
    
    if argumentsCount == 0 or argumentsCount>2:
        print("Usage:",sys.argv[0], "inputFile.mp4 [out.mp4]")
        exit()

    inputFileName = str(sys.argv[1])

    if argumentsCount == 2:
        outputFileName = str(sys.argv[2])
    else:
        outputFileName = "out.mp4"


    #load network configuration, weights and object names
    classes = None
    with open('coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    net = cv2.dnn.readNet('darknet/yolov3.weights', 'darknet/cfg/yolov3.cfg')

    processVideo(net, inputFileName, faceTrashold, outputFileName)

testOnVideo()