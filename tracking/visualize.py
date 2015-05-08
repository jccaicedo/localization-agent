import matplotlib as mpl
import matplotlib.animation
import matplotlib.pyplot as plt
import cv2
import os

import libDetection as libDet
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

def show_box(index, data, patch):
    print index, data.keys()
    topLeft = (data['boxes'][index][0], data['boxes'][index][1])
    width = data['boxes'][index][2]-data['boxes'][index][0]
    height = data['boxes'][index][3]-data['boxes'][index][1]
    patch.set_xy(topLeft)
    patch.set_width(width)
    patch.set_height(height)
    return patch

def view_memory(configPath, sequenceName):
    config.readConfiguration(configPath)
    imageSuffix = config.get('frameSuffix')
    sequenceDir = config.get('sequenceDir')
    testMemoryDir = config.get('testMemory')
    seqDatabasePath = config.get('testDatabase')
    seqDatabase = [x.strip() for x in open(seqDatabasePath, 'r')]
    
    if not sequenceName in seqDatabase:
        raise ValueError('{} not present in contents of {}'.format(seqName, seqDatabasePath))
    seqName, seqSpan, seqStart, seqEnd = cu.parseSequenceSpec(sequenceName)
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
            raise ValueError('Start {} or end {} outisde of bounds {},{}'.format(start, end, 1, len(aSequence.frames)))

    cv2.namedWindow(sequenceName)
    for frameIndex in range(start, end):
        aFrame = cv2.imread(os.path.join(aSequence.path, aSequence.frames[frameIndex]+imageSuffix))
        frameBbox = map(int, aSequence.boxes[frameIndex].tolist())
        gtFrame = aFrame.copy()
        cv2.rectangle(gtFrame, tuple(frameBbox[:2]), tuple(frameBbox[2:]), cv2.cv.CV_RGB(0,255,0))
        
        testMemoryPath = os.path.join(testMemoryDir, seqName, config.get('imageDir'), '{:04d}{}'.format(frameIndex, '.txt'))
        print testMemoryPath
        if os.path.exists(testMemoryPath):
            testMemory = cu.load_memory(os.path.join(testMemoryDir, seqName, config.get('imageDir'), '{:04d}{}'.format(frameIndex, '.txt')))
            for boxIndex in range(len(testMemory['boxes'])):
                if testMemory['actions'][boxIndex] == 8:
                    boxColor = cv2.cv.CV_RGB(0,0,255)
                else:
                    boxColor = cv2.cv.CV_RGB(255,0,0)
                    continue
                interactionBox = map(int, testMemory['boxes'][boxIndex])
                interactionFrame = gtFrame.copy()
                cv2.rectangle(interactionFrame, tuple(interactionBox[:2]), tuple(interactionBox[2:]), boxColor)
                cv2.imshow(sequenceName, interactionFrame)
                cv2.waitKey(30)
        else:
            cv2.imshow(sequenceName, gtFrame)
            cv2.waitKey(30)
