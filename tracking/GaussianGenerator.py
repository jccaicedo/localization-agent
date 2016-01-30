import sys
sys.path.insert(0, r'/home/fhdiaze/Code/localization-agent/tracking/')
import TrajectorySimulator as trsim
import numpy as np
import pickle

class GaussianGenerator(object):
    def __init__(self, seqLength=60, imageSize=100):
        self.imageSize = imageSize
        self.seqLength = seqLength
        trajectoryModelPath = '/home/fhdiaze/Code/localization-agent/notebooks/gmmDenseAbsoluteNormalizedOOT.pkl'
        # Generates a factory to create random simulator instances
        self.factory = trsim.SimulatorFactory(
            '/home/datasets/datasets1/mscoco', 
            trajectoryModelPath=trajectoryModelPath, 
            summaryPath='/home/datasets/datasets1/mscoco/cocoTrain2014Summary.pkl', 
            scenePathTemplate='images/train2014', objectPathTemplate='images/train2014'
            )
        modelFile = open(trajectoryModelPath, 'r')
        self.trajectoryModel = pickle.load(modelFile)
        modelFile.close()

    def getSimulator(self):
        emptyPolygon = True
        
        while emptyPolygon:
            simulator = self.factory.createInstance(drawBox=False, camera=True, drawCam=False, cameraContentTransforms=None, camSize=(self.imageSize, self.imageSize))
            emptyPolygon = len(simulator.polygon) == 0
            
        return simulator
    
    def getSingleSimulator(self):
        scenePath = "/home/datasets/datasets1/mscoco/images/train2014/COCO_train2014_000000011826.jpg"
        objectPath = "/home/datasets/datasets1/mscoco/images/train2014/COCO_train2014_000000250067.jpg"
        polygon = [618.23, 490.13, 615.76, 488.48, 612.89, 488.48, 610.42, 491.36, 609.19, 494.65, 607.54, 498.35, 607.13, 503.29, 606.72, 510.28, 610.42, 512.33, 612.89, 513.15, 616.18, 513.98, 619.88, 513.57, 621.93, 510.28, 623.58, 506.58, 623.58, 503.7, 623.16, 500.0, 621.93, 496.71, 619.46, 493.42]
        simulator = trsim.TrajectorySimulator(scenePath, objectPath, polygon=polygon, trajectoryModel=self.trajectoryModel, camSize=(self.imageSize, self.imageSize))
        
        return simulator

    def getBatch(self, batchSize):
        data = np.zeros((batchSize, self.seqLength, self.imageSize, self.imageSize), dtype=np.float32)
        label = np.zeros((batchSize, self.seqLength, 4))
        for i in range(batchSize):
            simulator = self.getSingleSimulator() 
            simulator.start()
            for j, frame in enumerate(simulator):
                data[i, j, :, :] = np.asarray(frame.convert('L'))
                label[i, j] = simulator.getBox()
                
        return data, label
