import matplotlib as mpl
import matplotlib.animation
import matplotlib.pyplot as plt
import cv2
import os

import utils.libDetection as libDet
import learn.rl.RLConfig as config
import utils.utils as cu
import sequence

def animate_memory(memoryDir, imageDir, gtPath, imageName, imageExtension='.jpg', memoryExtension='.txt'):
    frame = cv2.imread(os.path.join(imageDir, imageName + imageExtension))
    gt = load_gt(gtPath)
    data = load_memory(os.path.join(memoryDir, imageName + memoryExtension))
    plt.imshow(frame)
    figure = plt.gcf()
    currentAxis = plt.gca()
    patch = currentAxis.add_patch(mpl.patches.Rectangle((gt['0001'][0][0], gt['0001'][0][1]),gt['0001'][0][2]-gt['0001'][0][0],gt['0001'][0][3]-gt['0001'][0][1], fill=False, alpha=1))
    matplotlib.animation.FuncAnimation(figure, show_box, len(data['boxes']), fargs=(data, patch), interval=250)
    plt.show()

def animate_video(videoPath, fps=30):
    video = cv2.VideoCapture(videoPath)
    if not video.isOpened():
        raise Exception('Error opening video')
    success, frame = video.read()
    if not success:
        raise Exception('Error reading frame')
    axes = plt.imshow(frame)
    figure = plt.gcf()
    #subtract 2 frames as we have already read one and assuming 0-based indexing
    animation = matplotlib.animation.FuncAnimation(figure, play, int(video.get(cv2.CAP_PROP_FRAME_COUNT))-2, fargs=(video, axes), interval=int(1000/fps), repeat=False)
    return animation
    
def play(index, video, axes):
    success, frame = video.read()
    if not success:
        raise Exception('Error reading frame')
    axes.set_data(frame)
    return axes

def animate_subtractor(videoPath, fps=30):
    video = cv2.VideoCapture(videoPath)
    if not video.isOpened():
        raise Exception('Error opening video')
    success, frame = video.read()
    if not success:
        raise Exception('Error reading frame')
    subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    foreground = cv2.cvtColor(subtractor.apply(frame, learningRate=0.001), cv2.COLOR_GRAY2RGB)
    axes = plt.imshow(foreground)
    figure = plt.gcf()
    #subtract 2 frames as we have already read one and assuming 0-based indexing
    animation = matplotlib.animation.FuncAnimation(figure, play_subtractor, int(video.get(cv2.CAP_PROP_FRAME_COUNT))-2, fargs=(video, axes, subtractor), interval=int(1000/fps), repeat=False)
    return animation

def play_subtractor(index, video, axes, subtractor):
    success, frame = video.read()
    if not success:
        raise Exception('Error reading frame')
    foreground = cv2.cvtColor(subtractor.apply(frame, learningRate=0.001), cv2.COLOR_GRAY2RGB)
    axes.set_data(foreground)
    return axes

def animate_meanshift(videoPath, fps=30):
    video = cv2.VideoCapture(videoPath)
    if not video.isOpened():
        raise Exception('Error opening video')
    success, frame = video.read()
    if not success:
        raise Exception('Error reading frame')
    meanshift = cv2.pyrMeanShiftFiltering(frame, 5, 16)
    axes = plt.imshow(meanshift)
    figure = plt.gcf()
    #subtract 2 frames as we have already read one and assuming 0-based indexing
    animation = matplotlib.animation.FuncAnimation(figure, play_meanshift, int(video.get(cv2.CAP_PROP_FRAME_COUNT))-2, fargs=(video, axes), interval=int(1000/fps), repeat=False)
    return animation

def play_meanshift(index, video, axes):
    success, frame = video.read()
    if not success:
        raise Exception('Error reading frame')
    meanshift = cv2.pyrMeanShiftFiltering(frame, 5, 16)
    axes.set_data(meanshift)
    return axes

def show_box(index, data, patch):
    print index, data.keys()
    topLeft = (data['boxes'][index][0], data['boxes'][index][1])
    width = data['boxes'][index][2]-data['boxes'][index][0]
    height = data['boxes'][index][3]-data['boxes'][index][1]
    patch.set_xy(topLeft)
    patch.set_width(width)
    patch.set_height(height)
    return patch

PLACE_LANDMARK = 8

def view_memory(configPath, sequenceName, tofile=False, outputDir='/tmp'):
    config.readConfiguration(configPath)
    imageSuffix = config.get('frameSuffix')
    sequenceDir = config.get('sequenceDir')
    testMemoryDir = config.get('testMemory')
    seqDatabasePath = config.get('testDatabase')
    seqDatabase = [x.strip() for x in open(seqDatabasePath, 'r')]
    
    seqName, seqSpan, seqStart, seqEnd = cu.parseSequenceSpec(sequenceName)
    if not sequenceName in seqDatabase:
        raise ValueError('{} not present in contents of {}'.format(seqName, seqDatabasePath))
    imageDir = os.path.join(sequenceDir, seqName, config.get('imageDir'))
    gtPath = os.path.join(sequenceDir, seqName, config.get('gtFile'))
    aSequence = sequence.fromdir(imageDir, gtPath, suffix=imageSuffix)
    
    #no seqSpan means full sequence
    #frames start at 2 in list, but include 0 in gt
    if seqSpan is None:
        start = 0
        end = len(aSequence.frames)-1
    else:
        start = int(seqStart)
        end = int(seqEnd)
        if start < 1 or end >= len(aSequence.frames) or start > end:
            raise ValueError('Start {} or end {} outside of bounds {}-{}'.format(start, end, 1, len(aSequence.frames)))

    if not tofile:
        cv2.namedWindow(sequenceName)
    for frameIndex in range(start, end):
        aFrame = cv2.imread(os.path.join(aSequence.path, aSequence.frames[frameIndex]+imageSuffix))
        frameBbox = map(int, aSequence.boxes[frameIndex].tolist())
        gtFrame = aFrame.copy()
        cv2.rectangle(gtFrame, tuple(frameBbox[:2]), tuple(frameBbox[2:]), cv2.cv.CV_RGB(0,255,0))
        
        testMemoryPath = os.path.join(testMemoryDir, seqName, config.get('imageDir'), '{:04d}{}'.format(frameIndex+1, '.txt'))
        if os.path.exists(testMemoryPath):
            testMemory = cu.load_memory(testMemoryPath)
            for boxIndex in range(len(testMemory['boxes'])):
                if testMemory['actions'][boxIndex] == PLACE_LANDMARK:
                    boxColor = cv2.cv.CV_RGB(0,0,255)
                else:
                    boxColor = cv2.cv.CV_RGB(255,0,0)
                interactionBox = map(int, testMemory['boxes'][boxIndex])
                interactionFrame = gtFrame.copy()
                cv2.rectangle(interactionFrame, tuple(interactionBox[:2]), tuple(interactionBox[2:]), boxColor)
                if tofile:
                    outputPath = os.path.join(outputDir, seqName, '{:04d}_{}{}'.format(frameIndex+1, boxIndex, imageSuffix))
                    if not os.path.exists(os.path.dirname(outputPath)):
                        os.makedirs(os.path.dirname(outputPath))
                    cv2.imwrite(outputPath, interactionFrame)
                else:
                    cv2.imshow(sequenceName, interactionFrame)
                    cv2.waitKey(30)
        else:
            if tofile:
                outputPath = os.path.join(outputDir, seqName, '{:04d}{}'.format(frameIndex+1, imageSuffix))
                if not os.path.exists(os.path.dirname(outputPath)):
                    os.makedirs(os.path.dirname(outputPath))
                cv2.imwrite(outputPath, gtFrame)
            else:
                cv2.imshow(sequenceName, gtFrame)
                cv2.waitKey(30)
