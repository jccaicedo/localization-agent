import pickle
import numpy as NP
import os

class SimulatorDataManager(object):
    
    summaryPath = None
    objects = []
    imagesPath = None
    randGen = None
    SUMMARY_KEY = "summary"
    CATEGORY_KEY = "categories"
    
    def __init__(self, imagesPath, summaryPath):
        self.summaryPath = summaryPath
        self.imagesPath = imagesPath
     
        
    def getRandomObject(self):
        return NP.random.choice(self.objects)
    
    
    def setObjectsBySide(self, size, minSide):
        with open(self.summaryPath, "r") as summaryFile:
            summary = pickle.load(summaryFile)
        
        self.objects = []
        summary = [obj for obj in summary[self.SUMMARY_KEY] if self.checkObjectSide(obj["bbox"], minSide)]
        
        #Select a random image for the scene
        for obj in NP.random.choice(summary, size, replace=False):
            #Select a random image for the object
            objPath = os.path.join(self.imagesPath, obj["file_name"].strip())
            bbox = self.transformBBox(obj["bbox"])
            self.objects.append({"path":objPath, "segmentation":obj["segmentation"], "iscrowd":obj["iscrowd"], "bbox":bbox})
            
     
    def checkObjectSide(self, bboxCoco, minSide):
        # bboxCoco : [x,y,width,height]
        bbox = self.transformBBox(bboxCoco)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        return height >= minSide or width >= minSide
          
          
    def transformBBox(self, bboxCoco):
        # bboxCoco : [x,y,width,height]
        bbox = [bboxCoco[0], bboxCoco[1], bboxCoco[0] + bboxCoco[2], bboxCoco[1] + bboxCoco[3]]
        
        return bbox
    
    
    def saveObjects(self, outputPath):
        with open(outputPath, "wb") as objectsFile:
            pickle.dump(self.objects, objectsFile, pickle.HIGHEST_PROTOCOL)
            
            
    def loadObjects(self, objectsPath):
        with open(objectsPath, "r") as objectsFile:
            self.objects = pickle.load(objectsFile)