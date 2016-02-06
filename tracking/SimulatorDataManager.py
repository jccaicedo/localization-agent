import pickle
import random, time
import numpy as NP
import os

class SimulatorDataManager(object):
    
    summaryPath = None
    objects = []
    imagesPath = None
    randGen = None
    SUMMARY_KEY = 'summary'
    CATEGORY_KEY = 'categories'
    
    def __init__(self, imagesPath, summaryPath):
        self.summaryPath = summaryPath
        self.imagesPath = imagesPath
        self.randGen = random.Random()
        self.randGen.jumpahead(long(time.time()))
     
        
    def getRandomObject(self):
        return self.randGen.choice(self.objects)
    
    
    def setObjectsBySide(self, size, minSide):
        with open(self.summaryPath, 'r') as summaryFile:
            summary = pickle.load(summaryFile)
        
        self.objects = []
        summary = [obj for obj in summary[self.SUMMARY_KEY] if self.checkSegmentations(obj["segmentation"], minSide)]
        
        #Select a random image for the scene
        while len(self.objects) < size:
            #Select a random image for the object
            objData = self.randGen.choice(summary)
            polygon = self.randGen.choice([seg for seg in objData['segmentation'] if self.checkObjectSide(seg, minSide)])
            objPath = os.path.join(self.imagesPath, objData['file_name'].strip())
            self.objects.append({"path":objPath, "polygon":polygon})
      
    def checkSegmentations(self, segmentations, minSide):
        return any(self.checkObjectSide(seg, minSide) for seg in segmentations)
            
     
    def checkObjectSide(self, polygon, minSide):
        bbox = self.polygonBBox(polygon)
        heigh = bbox[0] - bbox[2]
        width = bbox[1] - bbox[3]
        
        return heigh >= minSide or width >= minSide
          
          
    def polygonBBox(self, polygon):
        '''Calculates the bounding box for the given polygon'''
        maskCoords = NP.array(polygon).reshape(len(polygon)/2,2).T
        bounds = map(int, (maskCoords[0].min(), maskCoords[1].min(), maskCoords[0].max(), maskCoords[1].max()))
        
        return bounds
    
    
    def saveObjects(self, outputPath):
        with open(outputPath, 'wb') as objectsFile:
            pickle.dump(self.objects, objectsFile, pickle.HIGHEST_PROTOCOL)
            
            
    def loadObjects(self, objectsPath):
        with open(objectsPath, 'r') as objectsFile:
            self.objects = pickle.load(objectsFile)