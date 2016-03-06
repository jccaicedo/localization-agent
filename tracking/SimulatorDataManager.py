import pickle
import numpy as NP
import os
import logging

class SimulatorDataManager(object):
    
    summaryPath = None
    SUMMARY_KEY = "summary"
    CATEGORY_KEY = "categories"
    
    def __init__(self, summaryPath):
        self.summaryPath = summaryPath
        
    
    """
    Correct image size to be even as needed by video codec.

    @type    size:   Integer
    @param   size:   The quantity of object to be returned. -1 if you want all
    @rtype:  {}
    @return: A dictionary with the objects, categories
    """ 
    def filterObjects(self, size, minSide, allowedCategories):
        with open(self.summaryPath, "r") as summaryFile:
            summary = pickle.load(summaryFile)
        
        outputObjects = []
        outputCategories = {}
        categories = summary[self.CATEGORY_KEY]
        summary = [obj for obj in summary[self.SUMMARY_KEY] if self.checkObjectSide(obj["segmentation"][0], minSide) and int(obj["category_id"]) in allowedCategories]
        
        size = len(summary) if size == -1 else size
        
        #Select a random image for the scene
        for obj in NP.random.choice(summary, size, replace=False):
            categoryId = int(obj["category_id"])
            obj["bbox"] = self.getBBox(obj["segmentation"][0])
            outputObjects.append(obj)
            outputCategories[categoryId] = categories[categoryId]
            
        outputSummary = {self.SUMMARY_KEY: outputObjects, self.CATEGORY_KEY: outputCategories}
        
        return outputSummary
            
    
    def filterSummary(self, size, minSide, allowedCategories, outputPath):
        outputSummary = self.filterObjects(size, minSide, allowedCategories)
        
        with open(outputPath, "w") as summaryFile:
            pickle.dump(outputSummary, summaryFile)
    
    def splitTrainVal(self, trainPercentage, minSide, allowedCategories, outputPathTrain, outputPathTest):
        filteredSummary = self.filterObjects(-1, minSide, allowedCategories)
        objs = filteredSummary[self.SUMMARY_KEY]
        cats = filteredSummary[self.CATEGORY_KEY]
        
        size = len(objs)
        train = int(size * trainPercentage / 100 + 0.5)
        
        logging.debug('Filtered summary size: %s ', size)
        logging.debug('Train summary size: %s ', train)
        logging.debug('Test summary size: %s ', size - train)
        
        trainObjs = objs[:train]
        testObjs = objs[train:]
        
        trainOutput = {self.SUMMARY_KEY: trainObjs,
                      self.CATEGORY_KEY: [ cats[catId] for catId in [obj["category_id"] for obj in trainObjs] ]}
        
        testOutput = {self.SUMMARY_KEY: testObjs,
                      self.CATEGORY_KEY: [ cats[catId] for catId in [obj["category_id"] for obj in testObjs] ]}
        
        with open(outputPathTrain, "w") as trainSummaryFile:
            pickle.dump(trainOutput, trainSummaryFile)
            
        with open(outputPathTest, "w") as testSummaryFile:
            pickle.dump(testOutput, testSummaryFile)
        
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