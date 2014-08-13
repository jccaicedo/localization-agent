import utils as cu
import libDetection as det
from dataProcessor import processData
import svmClassifier as svm
import latentSVM as lsvm

class Detector():
  def __init__(self,models,threshold,maxOverlap):
    self.models = models
    self.threshold = threshold
    self.maxOverlap = maxOverlap
    
  def run(self,img,features,bboxes):
    results = {}
    for category in models.keys():
      scores = self.models[category].predict(features)
      candIdx = scores>=self.threshold
      numCandidates = candIdx[candIdx==True].shape[0]
      if numCandidates > 0:
        candidateBoxes = [bboxes[t] for t in range(candIdx.shape[0]) if candIdx[t]]
        candidateScores = scores[candIdx]
        filteredBoxes,filteredScores = det.nonMaximumSuppression(candidateBoxes,candidateScores,self.maxOverlap)
        results[category] = (img,filteredBoxes,filteredScores)
    return results

categories = {'sofa':0,'train':0,'tvmonitor':0}

## Main Program Parameters
params = cu.loadParams("modelDir modelSuffix modelType testImageList featuresDir featuresExt maxOverlap threshold outputFile")
if params['modelType'] == 'latent':
  modelClass = lsvm.LatentSVM
elif params['modelType'] == 'linear':
  modelClass = svm.SVMDetector
else:
  import sys
  print 'Model not supported'
  sys.exit()

for c in categories.keys():
  filename = params['modelDir'] + '/' + c + params['modelSuffix']
  categories[c] = modelClass()
  categories[c].load(filename)

imageList = [x.replace('\n','') for x in open(params['testImageList'])]
maxOverlap = float(params['maxOverlap'])
threshold = float(params['threshold'])
## Run Detector
task = Detector(categories,threshold,maxOverlap)
result = processData(imageList,params['featuresDir'],params['featuresExt'],task)
# Prepare output files
for c in categories.keys():
  categories[c] = open(c + params['outputFile'],'w')
for data in result:
  for c in data.keys():
    img,filteredBoxes,filteredScores = data[c]
    for i in range(len(filteredBoxes)):
      b = filteredBoxes[i]
      categories[c].write(img + ' {:.8f} {:} {:} {:} {:}\n'.format(filteredScores[i],b[0],b[1],b[2],b[3]))
for c in categories.keys():
  categories[c].close()

