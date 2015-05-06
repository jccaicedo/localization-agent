import os,sys
import sklearn
import numpy as np
import utils as cu
import libLearning as learn
import libDetection as det
import maskDetector
import evaluation
import masks as mk
import warp as wp

## Program Variables
maxNegativeImages  = 200
maxVectorsCache    = 50000/(13*13)
maxNegativeVectors = 150000
np.random.seed( cu.randomSeed )

def mainLoop(modelType,modelArgs,positives,trainingList,featuresDir,featuresExt,modelOut,maxNegOverlap,iter):
  pos,posIdx,featSize,fmSize = positives
  featureSpace = pos.shape[1]
  startTime = cu.tic()
  if iter == 0:
    ## Random Negatives
    print ' >>> RANDOM NEGATIVES'
    N,negIdx = learn.getRandomNegs(featuresDir,trainingList,featuresExt,featSize,maxVectorsCache,maxNegativeImages)
    cellsPerImage = featSize/featureSpace
    N = N.reshape( (N.shape[0],featureSpace,fmSize,fmSize) ) # Recover original feature layout 
    neg = np.zeros( (cellsPerImage*N.shape[0],featureSpace) )
    for i in range(N.shape[0]):
      neg[i*cellsPerImage:(i+1)*cellsPerImage] = N[i].T.reshape((cellsPerImage,featureSpace)) # Unfold features
    hards = {'features':np.zeros((0,neg.shape[1])),'index':[]}
    lap = cu.toc('Random negatives matrix ('+str(neg.shape[0])+' instances)',startTime)
  else:
    ## Mine hard negatives
    print ' >>> MINING HARD NEGATIVES'
    model = det.createDetector(modelType,modelArgs)
    model.load(modelOut+'.'+ str( iter-1 ))
    detList,detMatrix = maskDetector.detectObjects(model,trainingList,featuresDir,featuresExt,-10.0)
    hdnList,detMatrix = maskDetector.selectHardNegatives(detList,detMatrix,posIdx,maxNegativeVectors)
    neg = maskDetector.loadHardNegativesFromMatrix(featuresDir,hdnList,detMatrix,featuresExt,featureSpace,maxNegativeVectors)
    hards = cu.loadMatrixNoCompression(modelOut+'.hards').item()
    lap = cu.toc('Hard negatives ('+str(neg.shape[0])+' mined + '+str(hards['features'].shape[0])+' previous instances)',startTime)

  ## Learn Detector
  neg = np.concatenate( (neg,hards['features']) )
  clf = det.createDetector(modelType,modelArgs)
  clf.learn(pos,neg)
  clf.save(modelOut+'.'+str(iter))
  lap = cu.toc('Classifier learned:',lap)

  ## Keep hard negatives for next iterations
  scores = clf.predict(neg)
  hardNegsIdx = np.argsort(scores)
  hardNeg = np.concatenate( (hards['features'], neg[hardNegsIdx[-cu.topHards:]]) )
  cu.saveMatrixNoCompression({'features':hardNeg},modelOut+'.hards')
  print ' ** Iteration',iter,'done'
#  return {'detector':clf,'pos':pos,'posIdx':posIdx,'neg':neg,'negIdx':negIdx}

########################################
## READ POSITIVE DATA
########################################
def readPositivesData(masksDir,featuresDir,featureExt,trainingPositives,category):
  ''' arch = mk.loadArchitecture('cnn.arch')
  #projections = mk.projectCoordsToReceptiveField(arch,'conv3')
  #convUnitToRegion = wp.Warping() '''
  
  posList = [x.split()[0] for x in open(trainingPositives)]
  pos,coor = None,{}
  featSize = 0
  for imName in posList:
    m = cu.loadMatrix(masksDir+'/'+imName+'.'+category)
    f,bboxes = cu.loadMatrixAndIndex(featuresDir+'/'+imName+'.'+featureExt)
    # Matrix dimensions
    regions,featSize = f.shape
    fmSize = m.shape[1]
    cells = fmSize*fmSize
    nfeat = f.shape[1]/cells
    # Transform Features Matrix
    f = f.reshape( (regions,nfeat,fmSize,fmSize) ) # Recover original feature layout 
    F = np.zeros( (regions*cells,nfeat) ) # Prepare new matrix layout
    M = np.zeros( (regions*cells) ) # Prepare labels vector 
    for i in range(regions):
        F[i*cells:(i+1)*cells] = f[i].T.reshape( (cells,nfeat) ) # Unfold features
        M[i*cells:(i+1)*cells] = m[i].T.reshape((cells)) # Transform Mask Matrices to Labels Vector
        ''' convUnitToRegion.prepare([0,0,226,226],map(float,bboxes[i][1:]))
        for x in range(fmSize):
            for y in range(fmSize):
                if m[i,x,y] == 1.0:
                    print imName,convUnitToRegion.transform(projections[x][y]) '''
    G = np.mgrid[0:regions,0:fmSize,0:fmSize].reshape((3,regions*cells)).T[:,[0,2,1]] # Grid of spatial locations
    # Get positives only
    P = F[M==1,:]
    coor[imName] = G[M==1,:]
    #if imName == '000815':
    #    print P[0][0:10],coor['000815'][0]
    if pos == None:
      pos = P
    else:
      pos = np.concatenate( (pos,P) )
  print 'Positive Matrix loaded ('+str(pos.shape[0])+' instances)'
  return (pos,coor,featSize,fmSize)

def parseModelParams(params):
  # Expected Format: k:v!k:v!
  params = params.split('!')
  result = {}
  for p in params:
    if p != '':
      k,v = p.split(':')
      result[k] = v
  return result

########################################
## MAIN PROGRAM
########################################
if __name__ == "__main__":
  params = cu.loadParams("modelType modelParams category positivesList trainingList masksDir featuresDir featuresExt modelOut overlap iterations")
  trainingList = [x.replace('\n','') for x in open(params['trainingList'])]
  maxNegOverlap = float(params['overlap'])
  iterations = int(params['iterations'])+1
  positives = readPositivesData(params['masksDir'],params['featuresDir'],params['featuresExt'],params['positivesList'],params['category'])
  args = parseModelParams(params['modelParams'])
  print " ++ LEARNING",params['modelType'],"MODEL WITH ARGS:",params['modelParams']," ++ "
  for i in range(iterations):
    mainLoop(params['modelType'],args,positives,list(trainingList),params['featuresDir'],params['featuresExt'],params['modelOut'],maxNegOverlap,i)
  os.system('rm '+params['modelOut']+'.hards')

