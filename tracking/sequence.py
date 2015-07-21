import os
import benchmarkUtils as benchutils
import time

class Sequence(object):
    
    def __init__(self):
        super(Sequence, self).__init__()
        self.frames = []
        self.boxes = []
        self.path = None
        self.source = None

    def __init__(gtPath, self):
        super(Sequence, self).__init__()
        self.boxes = benchutils.parse_gt(gtPath)

    def validate_lenghts(self):
        if not len(self.frames) == len(self.boxes):
            raise Exception('Number of frames ({}) and boxes ({}) do not match'.format(len(self.frames), len(self.boxes)))

def fromdir(dirPath, gtPath, suffix='.jpg'):
    aSequence = Sequence()
    aSequence.frames = sorted([framePath.replace(suffix, '') for framePath in os.listdir(dirPath) if framePath.endswith(suffix)])
    aSequence.path = dirPath
    aSequence.boxes = benchutils.parse_gt(gtPath)
    aSequence.validate_lenghts()
    return aSequence
  
try:

    import cv2

    def fromvideo(videoPath, gtPath):
        aSequence = Sequence(gtPath)
        aSequence.path = videoPath
        aVideo = cv2.VideoCapture(aSequence.path)
        if not aVideo.isOpened():
            raise Exception('Unable to open video {}'.format(aSequence.path))
        #0 based indexing
        aSequence.frames = map(str, range(int(aVideo.get(cv2.CAP_PROP_FRAME_COUNT))))
        aSequence.boxes = benchutils.parse_gt(gtPath)
        aSequence.validate_lenghts()
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

except:
    print 'Error importing cv2, dependend functions undefined'
