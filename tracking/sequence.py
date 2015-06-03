import cv2
import os
import benchmarkUtils as benchutils
import time

class Sequence(object):
    
    def __init__(self):
        super(Sequence, self).__init__()
        self.frames = []
        self.boxes = []
        self.path = None
        self.marker = 0

def fromdir(dirPath, gtPath, suffix='.jpg'):
    aSequence = Sequence()
    aSequence.frames = sorted([framePath.replace(suffix, '') for framePath in os.listdir(dirPath) if framePath.endswith(suffix)])
    aSequence.path = dirPath
    aSequence.marker = 0
    aSequence.boxes = benchutils.parse_gt(gtPath)
    return aSequence

def view(aSequence, winname='view', suffix='.jpg', fps=30):
    cv2.namedWindow(winname)
    period = 1000.0 / fps
    for aFrame in aSequence.frames:
        start = time.time()
        framePath = os.path.join(aSequence.path, aFrame+suffix)
        frame = cv2.imread(framePath)
        cv2.imshow(winname, frame)
        end = time.time()
        cv2.waitKey(int(period-(end - start)*1000))
    cv2.destroyWindow(winname)
    cv2.waitKey(1)
