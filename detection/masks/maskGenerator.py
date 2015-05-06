import os,sys
import Image
import masks as mk
from skimage import io
import libDetection as det
import numpy as np

def binaryMask(width,height,boxes):
    mask = np.zeros( (height, width) )
    for b in boxes:
        mask[ b[1]-1:b[3]-1, b[0]-1:b[2]-1 ] = 255
    return mask.astype(np.float32)

def createNegativeWindows(width, height, boxes, windowSize, stride):
    regions = []
    for j in range((height - windowSize) / stride):
        for i in range((width - windowSize) / stride):
            x1 = i * stride + 1
            y1 = j * stride + 1
            x2 = x1 + windowSize
            y2 = y1 + windowSize
            box = [x1,y1,x2,y2]
            maxOv = 0
            for b in boxes:
                ov1 = det.IoU(b, box)
                if ov1 > maxOv:
                    maxOv = ov1
            if maxOv >= 0.5:
                info = [1, maxOv]
            else:
                info = [0, maxOv]
            regions.append( info + box )
    return regions

def createPositiveWindows(width, height, boxes, windowSize, stride):
    windows = []
    winStrideRatio = float(stride)/float(windowSize)
    for b in boxes:
        b = map(int,b)
        w = b[2] - b[0]
        h = b[3] - b[1]
        if w > h:
            winSize = h
        else:
            winSize = w
        winStride = max(int(winSize*winStrideRatio),1)
        for j in range(max(b[1]-winStride,1), min(b[3] - winSize + winStride + 1, height), winStride):
            for i in range(max(b[0]-winStride,1), min(b[2] - winSize  + winStride + 1, width), winStride):
                x1 = max(i,1)
                y1 = max(j,1)
                x2 = min(x1 + winSize, width)
                y2 = min(y1 + winSize, height)
                box = [1, 1, x1, y1, x2, y2]
                windows.append(box)
    return windows

def makeWindowsFile(imgsDir, objects, windowSize, stride, outputFile, outputFigs=None):
    out = open(outputFile,'w')
    imgIdx = 0
    for name in objects.keys():
        img = Image.open(imgsDir+'/'+name+'.jpg')
        negative = createNegativeWindows(img.size[0],img.size[1], objects[name], windowSize, stride)
        positive = createPositiveWindows(img.size[0],img.size[1], objects[name], windowSize, stride)
        regions = positive + negative
        
        out.write('# ' + str(imgIdx) + '\n') #     # image_index 
        out.write(imgsDir+'/'+name+'.jpg\n') #     img_path
        out.write('3\n')                     #     channels 
        out.write(str(img.size[0])+'\n')     #     height 
        out.write(str(img.size[1])+'\n')     #     width
        out.write(str(len(regions))+'\n')    #     num_windows
        #     class_index overlap x1 y1 x2 y2
        for r in regions:
            out.write(' '.join(map(str,map(int,r)))+'\n')
            
        if outputFigs != None:
            #det.showDetections(imgsDir+'/'+name+'.jpg', objects[name], range(len(objects[name])), True, outputFile=outputFigs+'/'+name+'_boxes.png')
            winLabels = [x[2:] for x in regions if x[1] > 0.8]
            print name,'TargetObjects:',len(objects[name]),'PositiveWindows:',len(winLabels),'NegativesWindows:',len(regions)-len(winLabels)
            #if len(winLabels) > 0:
            #    det.showDetections(imgsDir+'/'+name+'.jpg', winLabels, range(len(winLabels)), fill=True, outputFile=outputFigs+'/'+name+'_win.png')
        imgIdx += 1                
    out.close()

if __name__ == "__main__":
    imgsDir = '/home/caicedo/data/allimgs/'

    # PASCAL 2007
    #groundTruthFile = '/home/caicedo/data/rcnn/lists/2007/trainval/all_gt_bboxes.txt'
    #outputFile = '/home/caicedo/data/rcnn/lists/2007/windows_pascal07_64_8.txt'
    #outputFigs = '/home/caicedo/data/rcnn/coveredObjects/'

    # PASCAL 2012 TrainVal
    #groundTruthFile = '/home/caicedo/data/PascalVOC/boxes/trainval/all_gt_boxes.txt'
    #outputFile = '/home/caicedo/data/rcnn/lists/2012/windows_pascal12_65_8.txt'
    #outputFigs = '/home/caicedo/data/rcnn/coveredObjects2012/'

    # PASCAL 2012 Train
    #groundTruthFile = '/home/caicedo/data/rcnn/lists/2012/train/all_gt_boxes.txt'
    #outputFile = '/home/caicedo/data/rcnn/lists/2012/train/windows_pascal12_65_8.txt'
    #outputFigs = '/home/caicedo/data/rcnn/coveredObjects2012/'

    # PASCAL 2012 Val
    groundTruthFile = '/home/caicedo/data/rcnn/lists/2012/val/all_gt_boxes.txt'
    outputFile = '/home/caicedo/data/rcnn/lists/2012/val/windows_pascal12_65_8.txt'
    outputFigs = '/home/caicedo/data/rcnn/coveredObjects2012/'

    objects = mk.loadBoxIndexFile(groundTruthFile)
    makeWindowsFile(imgsDir, objects, 65, 8, outputFile, outputFigs)
