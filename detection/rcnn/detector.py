import utils as cu
import libDetection as det
from dataProcessor import processData

class Detector():
  def __init__(self,model,threshold,maxOverlap):
    self.model = model
    self.threshold = threshold
    self.maxOverlap = maxOverlap
    
  def run(self,img,features,bboxes):
    scores,labels = self.model.predictAll(features,bboxes)
    candIdx = scores>=self.threshold
    numCandidates = candIdx[candIdx==True].shape[0]
    if numCandidates > 0:
      candidateBoxes = [bboxes[t]+[labels[t]] for t in range(candIdx.shape[0]) if candIdx[t]]
      candidateScores = scores[candIdx]
      filteredBoxes,filteredScores = det.nonMaximumSuppression(candidateBoxes,candidateScores,self.maxOverlap)
      return (img,filteredBoxes,filteredScores)
    else:
      return None

########################################
## RUN OBJECT DETECTOR
########################################
def detectObjects(model,imageList,featuresDir,featuresExt,maxOverlap,threshold,outputFile=None):
  task = Detector(model,threshold,maxOverlap)
  result = processData(imageList,featuresDir,featuresExt,task)
  if outputFile != None:
    outf = open(outputFile,'w')
    writeF = lambda x,y,b: outf.write(x + ' {:.8f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f}\n'.format(y,b[0],b[1],b[2],b[3],b[4]))
  else:
    writeF = lambda x,y,b: x
  detectionsList = []
  for data in result:
    img,filteredBoxes,filteredScores = data
    for i in range(len(filteredBoxes)):
      b = filteredBoxes[i]
      writeF(img,filteredScores[i],b)
      detectionsList.append( [img,filteredScores[i],b[0],b[1],b[2],b[3],b[4]] )
  if outputFile != None:
    outf.close()
  return detectionsList

########################################
## MAIN PROGRAM
########################################
if __name__ == "__main__":
  ## Main Program Parameters
  params = cu.loadParams("modelType modelFile testImageList featuresDir featuresExt maxOverlap threshold outputFile")
  model = det.createDetector(params['modelType'])
  model.load(params['modelFile'])
  imageList = [x.replace('\n','') for x in open(params['testImageList'])]
  maxOverlap = float(params['maxOverlap'])
  threshold = float(params['threshold'])
  detectObjects(model,imageList,params['featuresDir'],params['featuresExt'],maxOverlap,threshold,params['outputFile'])

