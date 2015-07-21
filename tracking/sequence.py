import os
import benchmarkUtils as benchutils
import time
import numpy
    
DIR_SOURCE = 0
VIDEO_SOURCE = 1

class Sequence(object):

    def __init__(self, gtPath=None):
        super(Sequence, self).__init__()
        self.frames = []
        self.boxes = []
        if gtPath is not None and os.path.exists(gtPath):
            self.boxes = benchutils.parse_gt(gtPath)
        self.path = None
        self.source = None

    def validate_lenghts(self):
        if len(self.boxes) == 0:
            print 'Empty boxes, not validating'
            return
        if not len(self.frames) == len(self.boxes):
            raise Exception('Number of frames ({}) and boxes ({}) do not match'.format(len(self.frames), len(self.boxes)))

def fromdir(dirPath, gtPath, suffix='.jpg'):
    aSequence = Sequence(gtPath)
    aSequence.frames = sorted([framePath.replace(suffix, '') for framePath in os.listdir(dirPath) if framePath.endswith(suffix)])
    aSequence.path = dirPath
    aSequence.validate_lenghts()
    aSequence.source = DIR_SOURCE
    return aSequence
  
try:

    import cv2

    def fromvideo(videoPath, gtPath):
        aSequence = Sequence(gtPath)
        aSequence.path = videoPath
        video = cv2.VideoCapture(aSequence.path)
        if not video.isOpened():
            raise Exception('Unable to open video {}'.format(aSequence.path))
        #0 based indexing
        aSequence.frames = map(str, range(int(video.get(cv2.CAP_PROP_FRAME_COUNT))))
        aSequence.validate_lenghts()
        aSequence.source = VIDEO_SOURCE
        video.release()
        return aSequence

    def view(aSequence, winname='view', suffix='.jpg', fps=30):
        if aSequence.source == VIDEO_SOURCE:
            video = cv2.VideoCapture(aSequence.path)
            if not video.isOpened():
                raise Exception('Unable to open video {}'.format(aSequence.path))
            frameIndex = 0
            videoFps = video.get(cv2.CAP_PROP_FPS)
            if not numpy.isnan(videoFps):
                fps = videoFps
        cv2.namedWindow(winname)
        period = 1000.0 / fps
        for aFrame in aSequence.frames:
            start = time.time()
            if aSequence.source == DIR_SOURCE:
                framePath = os.path.join(aSequence.path, aFrame+suffix)
                frame = cv2.imread(framePath)
            else:
                success, frame = video.read()
                if not success:
                    raise Exception('Error reading frame {}'.format(frameIndex))
                frameIndex += 1
            cv2.imshow(winname, frame)
            end = time.time()
            cv2.waitKey(int(period-(end - start)*1000))
        if aSequence.source == VIDEO_SOURCE:
            video.release()
        cv2.destroyWindow(winname)
        cv2.waitKey(1)

except:
    print 'Error importing cv2, dependend functions undefined'
