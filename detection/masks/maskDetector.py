import utils as cu
import libDetection as det
from dataProcessor import processData
import numpy as np
import masks as mk
import warp as wp
import dataProcessor as dp

########################################
## MASK DETECTOR
########################################
# This detector assumes a linear classifier. Does not work with other models (e.g. LatentSVMs)
class MaskDetector():
  def __init__(self,model,threshold,projector=None):
    self.model = model
    self.threshold = threshold
    self.featureSpace = model.clf.coef_.shape[1]
    self.projector = projector
    self.count = 0
    
  def run(self,img,features,bboxes):
    regions = features.shape[0]
    cells = features.shape[1]/self.featureSpace
    fmSize = int(np.sqrt(cells))
    features = features.reshape( (regions,self.featureSpace,fmSize,fmSize) ) # Recover original feature layout 
    F = np.zeros( (regions*cells, self.featureSpace) )
    for i in range(regions): 
        F[i*cells:(i+1)*cells] = features[i].T.reshape( (cells,self.featureSpace) ) # Unfold features
    scores = self.model.predict(F)
    s = np.sqrt(scores.shape[0]/features.shape[0])
    grid = np.mgrid[0:regions,0:fmSize,0:fmSize].reshape((3,regions*cells)).T[:,[0,2,1]] # Grid of spatial locations
    candidates = scores > self.threshold
    R = np.zeros( (len(candidates[candidates==True]),5) )
    R[:,0] = 1
    R[:,1:4] = grid[candidates,:]
    R[:,4] = scores[candidates]
    if self.projector == None:
        return (img,R)
    else:
        return self.projector.project(img, bboxes, R)

def detectObjects(model,imageList,featuresDir,featuresExt,threshold,projector=None):
  task = MaskDetector(model,threshold,projector)
  result = processData(imageList,featuresDir,featuresExt,task)
  if projector == None:
      totalDetections = reduce(lambda x,y:x+y,[d[1].shape[0] for d in result])
      detections = np.zeros( (totalDetections,5) )
      images = {}
      imgId = 0
      i = 0
      for data in result:
          img,cells = data
          cells[:,0] = cells[:,0]*imgId
          detections[i:i+cells.shape[0],:] = cells
          images[img] = imgId
          imgId += 1
          i = i + cells.shape[0]
      return (images,detections[0:i,:])
  else:
      resultsList = []
      for data in result:
          resultsList += data
      return resultsList 

########################################
## RANK DETECTIONS AND PICK TOP NEGATIVES
######################################## 
def selectHardNegatives(images,detections,groundTruth,maxNegs):
    print 'Selecting Hard Negatives from a list of ',len(detections)
    rank = (-detections[:,4]).argsort()
    detections = np.insert(detections[ rank ],0,range(len(detections)),axis=1)
    delete = []
    for k in groundTruth.keys():
        if len(groundTruth[k]) == 0:
            continue
        imgDet = detections[ detections[:,1] == images[k] ]
        gt = groundTruth[k]
        for i in range(len(gt)):
            candidatesA = imgDet[imgDet[:,2]==gt[i][0]]
            if len(candidatesA) == 0: continue
            candidatesB = candidatesA[candidatesA[:,3]==gt[i][1]]
            if len(candidatesB) == 0: continue
            candidatesC = candidatesB[candidatesB[:,4]==gt[i][2]]
            if len(candidatesC) == 0: continue
            idx = candidatesC[:,0]
            for j in idx:
                delete.append(j)
    print len(delete),'deleted from negative set'
    mask = np.ones(len(detections),dtype=bool)
    mask[delete] = False
    detections = detections[mask]
    return (images,detections[0:maxNegs])

###############################################
# Load Hard Negatives from predefined list
###############################################
class LoadHardNegatives():
  def __init__(self,imagesIdx,detMatrix,featureSpace):
    self.imagesIdx = imagesIdx
    self.detMatrix = detMatrix
    self.featureSpace = featureSpace

  def run(self,img,features,bboxes):
    dets = self.detMatrix[ self.detMatrix[:,1] == self.imagesIdx[img] ]
    rows = len(dets)
    result = np.zeros( (rows,self.featureSpace) )
    for i in range(rows):
      idx,im,x,y,z,s = dets[i]
      result[i,:] = features[ x, self.featureSpace*y*z:self.featureSpace*y*z+self.featureSpace ]
    return result

def loadHardNegativesFromMatrix(featuresDir,imagesIdx,detMatrix,featuresExt,numFeatures,totalNegatives):
  i = 0
  task = LoadHardNegatives(imagesIdx,detMatrix,numFeatures)
  result = dp.processData(imagesIdx.keys(),featuresDir,featuresExt,task)
  hardng = cu.emptyMatrix([totalNegatives,numFeatures])
  while len(result) > 0:
    data = result.pop(0)
    hardng[i:i+data.shape[0],:] = data
    i = i + data.shape[0]
  return hardng[0:i,:]

###############################################
# Projecting Local Predictions to Image Plane
###############################################  
def selectProposalsByArea(proposals,minArea,maxArea):
    areas = [det.area(box) for box in proposals]
    proposals = [[0]+proposals[i] for i in range(len(areas)) if areas[i] >= minArea and areas[i] < maxArea]
    return proposals

def scoreProposals(scoringAreas,imgProp):
    propScores = []
    for box in imgProp:
        boxScore = 0.0
        for sarea in scoringAreas:
            boxScore += det.IoU(box[1:], sarea[0:4])*sarea[4]
        propScores.append(boxScore)
    return propScores

class PredictionsToImagePlane():
    def __init__(self,proposals,convLayer,minArea,maxArea,nmsThreshold):
        arch = mk.loadArchitecture('cnn.arch')
        self.projections = mk.projectCoordsToReceptiveField(arch,convLayer)
        self.proposals = proposals
        self.minArea = minArea
        self.maxArea = maxArea
        self.nmsThreshold = nmsThreshold
        
    def project(self,img,regions,imgDet):
        imgProp = selectProposalsByArea(self.proposals[img],self.minArea,self.maxArea)
        results = []
        transformedAreas,areasScores = [],[]
        convUnitToRegion = wp.Warping()
        print 'Image:',img,'proposals:',len(imgProp),'detections:',len(imgDet)
        for r in range(len(regions)):
            regionDet = imgDet[imgDet[:,1]==r]
            convUnitToRegion.prepare([0,0,226,226],map(float,regions[r][1:]))
            for d in range(len(regionDet)):
                x,y = regionDet[d,2:4]
                sa = convUnitToRegion.transform(self.projections[int(x)][int(y)])
                transformedAreas.append( [0] + sa )
                areasScores.append(regionDet[d,4])
        transformedAreas,areasScores = det.nonMaximumSuppression(transformedAreas, areasScores, 0.5)
        #self.visualizeScoreMaps(img,transformedAreas, areasScores)
        transformedAreas = [transformedAreas[i]+[areasScores[i]] for i in range(len(areasScores))]
        propScores = scoreProposals(transformedAreas,imgProp)
        imgProp = [imgProp[i] for i in range(len(imgProp)) if propScores[i] >= 0.0]
        propScores = [propScores[i] for i in range(len(propScores)) if propScores[i] >= 0.0]
        finalBoxes,finalScores = det.nonMaximumSuppression(imgProp, propScores, self.nmsThreshold)
        for i in range(len(finalBoxes)):
            results.append( [img,finalScores[i]] + map(int,finalBoxes[i]) )
        return results
    
    def visualizeScoreMaps(self, image, areas, scores):
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        from matplotlib.patches import Rectangle
        import Image
        imgsDir = '/home/caicedo/data/allimgs/'
        
        fig = plt.figure()
        img = mpimg.imread(imgsDir+image+'.jpg')
        (ox,oy,oc) = img.shape
        sp = fig.add_subplot(1,2,1)
        imgplot = plt.imshow(img,origin='lower')
                 
        smap = np.zeros( (ox,oy) )
        for i in range(len(areas)):
            a = map(int,areas[i])
            smap[ a[1]:a[3], a[0]:a[2] ] += scores[i]
            
        a = fig.add_subplot(1,2,2)
        smap[0,0] = 20
        smap[ox-1,oy-1] = -20
        plt.imshow(smap)
        plt.savefig('/home/caicedo/data/rcnn/masksOut/'+image+'.png',bbox_inches='tight')        

########################################
## MAIN PROGRAM
########################################
if __name__ == "__main__":
    MIN_AREA = 99.0*99.0
    MAX_AREA = 227.0*227.0
    CONV_LAYER = 'conv3'
    ## Main Program Parameters
    params = cu.loadParams("modelFile testImageList proposalsFile featuresDir featuresExt threshold outputDir")
    model = det.createDetector('linear')
    model.load(params['modelFile'])
    imageList = [x.replace('\n','') for x in open(params['testImageList'])]
    proposals = mk.loadBoxIndexFile(params['proposalsFile'])
    threshold = float(params['threshold'])
    ## Make detections and transfer scores
    projector = PredictionsToImagePlane(proposals,CONV_LAYER,MIN_AREA,MAX_AREA,0.7)
    results = detectObjects(model,imageList,params['featuresDir'],params['featuresExt'],-10.0,projector)
    out = open(params['outputDir'],'w')
    for r in results:
        out.write(' '.join(map(str,r))+' 0\n')
    out.close()
