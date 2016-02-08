import pickle
import numpy as NP
import os

class SimulatorDataManager(object):
    
    summaryPath = None
    SUMMARY_KEY = "summary"
    CATEGORY_KEY = "categories"
    
    def __init__(self, summaryPath):
        self.summaryPath = summaryPath
     
        
    def getRandomObject(self):
        return NP.random.choice(self.objects)
    
    
    def exportObjectsBySide(self, size, minSide, outputPath):
        with open(self.summaryPath, "r") as summaryFile:
            summary = pickle.load(summaryFile)
        
        outputObjects = []
        outputCategories = {}
        categories = summary[self.CATEGORY_KEY]
        summary = [obj for obj in summary[self.SUMMARY_KEY] if self.checkObjectSide(obj["bbox"], minSide)]
        
        #Select a random image for the scene
        for obj in NP.random.choice(summary, size, replace=False):
            categoryId = int(obj['category_id'])
            outputObjects.append(obj)
            outputCategories[categoryId] = categories[categoryId]
            
        outputSummary = {self.SUMMARY_KEY: outputObjects, self.CATEGORY_KEY: outputCategories}
        
        with open(outputPath, 'w') as summaryFile:
            pickle.dump(outputSummary, summaryFile)
        
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