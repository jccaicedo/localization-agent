import utils as cu
import libDetection as det
from dataProcessor import processData

class Checker():
  def __init__(self):
   print 'Starting checker' 

  def run(self,img,features,bboxes):
    return img,features.shape[0] == len(bboxes)

## Main Program Parameters
params = cu.loadParams("testImageList featuresDir featuresExt")

imageList = [x.replace('\n','') for x in open(params['testImageList'])]
## Run Detector
task = Checker()
start = cu.tic()
result = processData(imageList,params['featuresDir'],params['featuresExt'],task)
cu.toc('All images checked',start)
totalP = 0
for data in result:
  img,r = data
  if not r:
    print 'Problems with',img
    totalP += 1
print 'Total problems:',totalP
