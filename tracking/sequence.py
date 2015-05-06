import cv2
import os
import benchmarkUtils as benchutils

class Sequence(object):
    
    def __init__(self):
        self.frames = []
        self.boxes = {}
        self.path = None
        self.marker = 0
        super(Sequence, self).__init__()

def fromdir(dirPath, gtPath, suffix='.jpg'):
    aSequence = Sequence()
    aSequence.frames = sorted([framePath for framePath in os.listdir(dirPath) if framePath.endswith(suffix)])
    aSequence.path = dirPath
    aSequence.marker = 0
    aSequence.boxes[gtPath] = benchutils.parse_gt(gtPath)
    return aSequence

def view(aSequence, winname='view'):
    cv2.namedWindow(winname)
    for aFrame in aSequence.frames:
        framePath = os.path.join(aSequence.path, aFrame)
        frame = cv2.imread(framePath)
        cv2.imshow(winname, frame)
        cv2.waitKey(30)
    cv2.destroyWindow(winname)
    cv2.waitKey(1)
