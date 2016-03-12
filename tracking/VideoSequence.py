# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 09:38:35 2015

@author: Fredy
"""
import subprocess
import numpy as np
import tempfile
import shutil
import os
from PIL import Image, ImageDraw
from VideoSequenceData import boxToPolygon
import csv

def displayHTML(output):
    videoSrc = 'data:video/mp4;base64,' + open(output, 'rb').read().encode('base64')
    videoTag = '<video controls width=\"320\" height=\"240\"><source src=\"{0}\" type=\"video/mp4\">Unsupported tag</video>'
    return videoTag.format(videoSrc)

def fromarray(data):
    return VideoSequence((Image.fromarray(frame.astype(np.uint8)) for frame in data))

class VideoSequence:
    PROCESS_TEMPLATE = 'avconv -y -f image2pipe -vcodec mjpeg -r {} -i - -vcodec libx264 -qscale 5 -r {} {}'
    PROCESS_TEMPLATE_OFFLINE = 'avconv -y -f image2 -vcodec mjpeg -r {} -i {} -vcodec libx264 -qscale 5 -r {} {}'
    
    """
    Create a Sequence base on a list of frames.

    @type  frames: iterator(PIL.Image)
    @param frames: The frames iterator
    """
    def __init__(self, frames):
        self.frames = frames
        self.boxes = {}
    
    """
    Add bounding boxes to the frames.

    @type  boxes:  [[number]]
    @param boxes:  A list of list. Each element must be a list containing the points of a polygon.
    @type  outline: string
    @param outline: The color for the boxes.
    """
    def addBoxes(self, boxes, outline):
        # add the new bounding boxes to the dictionary
        invalidBoxIndexes = [index for index, boxLength in enumerate(map(len,boxes)) if not (boxLength == 4 or (boxLength >= 6 and boxLength % 2 == 0))]
        if len(invalidBoxIndexes) > 0:
            raise Exception('Invalid boxes at indexes: {}'.format(invalidBoxIndexes))
        self.boxes[outline] = boxes

    
    """
    Return the frames with bounding boxes drawn.

    @rtype:  iterator(PIL.Image)
    @return: An iterator over the frames.
    """
    def getFramesWithBoxes(self):
        for index, frame in enumerate(self.frames, start=0):
            frameBoxes = [(outline, boxes[index]) for outline, boxes in self.boxes.items()]
            self.plotBoxes(frame, frameBoxes)
            frame = self.resizeImage(frame)
            yield frame
    
    
    """
    Plot many bounding boxes in a frame.

    @type    frame:           PIL.Image
    @param   frame:           The frame
    @type    outlineBoxPairs: [(string, [])]
    @param   outlineBoxPairs: The list of color, points of a polygon
    """ 
    def plotBoxes(self, frame, outlineBoxPairs):
        for outline, box in outlineBoxPairs:
            self.plotBox(frame, box, outline)

    
    """
    Plot a bounding box in a frame.

    @type    frame:   PIL.Image
    @param   frame:   The frame
    @type    box:     []
    @param   box:     The list of points of a box
    @type    outline: string
    @param   outline: The name of the bounding box color.
    """ 
    def plotBox(self, frame, box, outline):
        draw = ImageDraw.Draw(frame)
        dataList = list(box)
        if len(dataList) == 4:
            draw.rectangle(dataList, outline=outline)
        elif len(dataList) >= 6 and len(dataList) % 2 == 0:
            draw.polygon(dataList, outline=outline)
        else:
            raise Exception('Unrecognized box format: {}'.format(dataList))
        #TODO: add missing point in case of rectangle
        for i in range(len(dataList)/2):
            draw.text(dataList[2*i:2*i+2], str(i), fill=outline)
    
    
    """
    Correct image size to be even as needed by video codec.

    @type    image:   PIL.Image
    @param   image:   The frame
    @rtype:  PIL.image
    @return: The resized image
    """ 
    def resizeImage(self, image): # 
        evenSize = list(image.size)
        resize = False
        
        for index in range(len(evenSize)):
            if evenSize[index] % 2 == 1:
                evenSize[index] += 1
                resize = True
        
        if(resize):
            evenSize = tuple(evenSize)
            image = image.resize(evenSize, Image.ANTIALIAS)
        
        return image
    
    
    """
    Export the sequence to a video.

    @type    fps:    number
    @param   fps:    Frames per second
    @type    output: string
    @param   output: The name of the output video.
    """ 
    def exportToVideoPiped(self, fps, output):
        conversionProcess = subprocess.Popen(self.PROCESS_TEMPLATE.format(fps, fps, output).split(' '), stdin=subprocess.PIPE)
        
        for frame in self.getFramesWithBoxes():
            frame.save(conversionProcess.stdin, 'JPEG')

        conversionProcess.stdin.close()
        conversionProcess.wait()
        
    """
    Export the sequence to a video.

    @type    fps:    number
    @param   fps:    Frames per second
    @type    output: string
    @param   output: The name of the output video.
    """ 
    def exportToVideo(self, fps, output, keep=False):
        if keep:
            outputPath = os.path.dirname(output)
            os.makedirs(outputPath)
        else:
            outputPath = tempfile.mkdtemp()
        processString = self.PROCESS_TEMPLATE_OFFLINE.format(fps, os.path.join(outputPath, '%08d.jpg'), fps, output)
        
        for index, frame in enumerate(self.getFramesWithBoxes(), start=0):
            frame.save(os.path.join(outputPath, '{:08d}.jpg'.format(index)), format='JPEG')
        
        conversionProcess = subprocess.Popen(processString.split(' '), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        conversionProcess.wait()
        if not keep:
            shutil.rmtree(outputPath)

    def exportBoxes(self, output, outline):
        polygons = self.boxes[outline]
        with open(output, 'w') as csvFile:
            csvwriter = csv.writer(csvFile)
            if len(polygons[0]) == 4:
                converted = []
                for box in polygons:
                    csvwriter.writerow(boxToPolygon(box))
            else:
                for polygon in polygons:
                    csvwriter.writerow(polygon)
