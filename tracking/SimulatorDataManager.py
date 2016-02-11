import pickle
import numpy as NP
import os

class SimulatorDataManager(object):
    
    summaryPath = None
    SUMMARY_KEY = "summary"
    CATEGORY_KEY = "categories"
    
    def __init__(self, summaryPath):
        self.summaryPath = summaryPath
        
    
    def filterObjects(self, size, minSide, allowedCategories, outputPath):
        with open(self.summaryPath, "r") as summaryFile:
            summary = pickle.load(summaryFile)
        
        outputObjects = []
        outputCategories = {}
        categories = summary[self.CATEGORY_KEY]
        summary = [obj for obj in summary[self.SUMMARY_KEY] if self.checkObjectSide(obj["segmentation"][0], minSide) and int(obj['category_id']) in allowedCategories]
        
        #Select a random image for the scene
        for obj in NP.random.choice(summary, size, replace=False):
            categoryId = int(obj['category_id'])
            obj["bbox"] = self.getBBox(obj["segmentation"][0])
            outputObjects.append(obj)
            outputCategories[categoryId] = categories[categoryId]
            
        outputSummary = {self.SUMMARY_KEY: outputObjects, self.CATEGORY_KEY: outputCategories}
        
        with open(outputPath, 'w') as summaryFile:
            pickle.dump(outputSummary, summaryFile)
        
    def checkObjectSide(self, polygon, minSide):
        # bboxCoco : [x, y, x,y, ...]
        bbox = self.getBBox(polygon)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        return height >= minSide or width >= minSide
          
          
    def getBBox(self, polygon):
        '''Calculates the bounding box for the given polygon'''
        maskCoords = NP.array(polygon).reshape(len(polygon)/2,2).T
        bounds = map(int, (maskCoords[0].min(), maskCoords[1].min(), maskCoords[0].max(), maskCoords[1].max()))
        return bounds