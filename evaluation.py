import os,sys
import re
import utils as cu
import libDetection as ldet

#Functions
def cumsum(a):
  b = [0.0 for x in range(0,len(a))]
  for i in range(0,len(a)):
    b[i] = b[i-1]+a[i]
  return b

intersect = lambda x,y: [max(x[0],y[0]),max(x[1],y[1]),min(x[2],y[2]),min(x[3],y[3])]

area = lambda x: (x[2]-x[0]+1)*(x[3]-x[1]+1)

def loadGroundTruthAnnotations(indexData):
  groundTruth = dict()
  for gt in indexData:
    imgName = re.sub(r'(.+/){0,1}(.+).jpg',r'\2',gt[0])
    data = map(float,gt[1:]) + [False]
    try:
      groundTruth[imgName].append( data )
    except:
      groundTruth[imgName] = [ data ]
 
  return groundTruth

def loadDetections(detectionsData):
  detections = list()
  for d in detectionsData:
    if d[0].endswith('.jpg'):
      d[0] = re.sub(r'(.+/){0,1}(.+).jpg',r'\2',d[0])
    data = [d[0], float(d[1])] + map(float,d[2:])
    #if float(d[1]) < -1.0: continue
    detections.append(data)
  # Sort Detections by decreasing confidence
  detections.sort(key=lambda x:x[1], reverse=True)
  print 'Detections:',len(detections)
  return detections

def loadMaxDetectionsPerImage(detectionsData,maxDetections):
  imageCounts = {}
  for d in detectionsData:
    if d[0].endswith('.jpg'):
      d[0] = re.sub(r'(.+/){0,1}(.+).jpg',r'\2',d[0])
    data = [d[0], float(d[1])] + map(float,d[2:])
    #if float(d[1]) < -2.0: continue
    try: imageCounts[d[0]].append(data)
    except: imageCounts[d[0]] = [data]

  detections = list()
  for i in imageCounts.keys():
    imageCounts[i].sort(key=lambda x:x[1], reverse=True)
    for j in range( min(maxDetections,len(imageCounts[i])) ):
      detections.append(imageCounts[i][j])
  # Sort Detections by decreasing confidence
  detections.sort(key=lambda x:x[1], reverse=True)
  print 'Detections:',len(detections)
  return detections


def evaluateDetections(groundTruth,detections,minOverlap,outFile=None,overlapMeasure=ldet.IoU,allowDuplicates=False):
  print 'Minimum overlap:',minOverlap
  if outFile != None:
    log   = open(outFile+'.log','w')
    paste = lambda x,y:str(x)+" "+str(y)
    mix   = lambda det: reduce(paste, map(int,det[2:]))
    logW  = lambda det,maxOverlap,label: log.write(det[0]+' '+mix(det)+' '+str(maxOverlap)+' '+label+'\n')
  else:
    logW = lambda x,y,z: x
  logData = []
  # Assign detections to ground truth objects
  tp = [0 for x in range(0,len(detections))]
  fp = [0 for x in range(0,len(detections))]
  for i in range(0,len(detections)):
    det = detections[i]
    maxOverlap = -1
    index = -1
    label = "0"
    if det[0] in groundTruth.keys():
      for j in range(0,len(groundTruth[det[0]])):
        bbox = groundTruth[det[0]][j]
        #overlap = ldet.IoU(bbox[0:4],det[2:6])
        #overlap = ldet.overlap(det[2:6],bbox[0:4])
        overlap = overlapMeasure(det[2:6],bbox[0:4])
        if overlap > maxOverlap:
          maxOverlap = overlap
          index = j
      if maxOverlap >= minOverlap:
        if not groundTruth[det[0]][index][4]: # Has not been assigned
          tp[i] = 1 # True Positive
          if not allowDuplicates:
            groundTruth[det[0]][index][4] = True
          label = "1"
        else:
          fp[i] = 1 # False Positive
      else:
        fp[i] = 1 # False Positive
    else:
      fp[i] = 1 # False Positive
    logW(det,maxOverlap,label)
    logData.append( [det[0]]+map(int,det[2:])+[maxOverlap,label] )

  if outFile != None:
    missedF = open(outFile+'.missed','w')
    for k in groundTruth.keys():
      for det in groundTruth[k]:
        if not det[4]:
          missedF.write(k+' {:} {:} {:} {:}\n'.format(det[0],det[1],det[2],det[3]))
    missedF.close()
    log.close()
  return {'log':logData,'fp':fp,'tp':tp}

def computePrecisionRecall(numPositives,tp,fp,outFile):
  # Compute Precision/Recall
  numTP = sum(tp)
  numFP = sum(fp)
  print "True Positives:",numTP,"False Positives:",numFP
  if numTP > numPositives:
    totalPositives = numTP
  else:
    totalPositives = numPositives
  tp = cumsum(tp)
  fp = cumsum(fp)
  recall = map(lambda x:x/float(totalPositives), tp)
  precision = [tp[i]/(tp[i]+fp[i]) for i in range(0,len(tp))]
  output = open(outFile,"w")
  for i in range(0,len(recall)):
    output.write(str(recall[i])+" "+str(precision[i])+"\n")
  
  '''
  PASCAL VOC 2012 devkit
  mrec=[0 ; rec ; 1];
  mpre=[0 ; prec ; 0];
  for i=numel(mpre)-1:-1:1
    mpre(i)=max(mpre(i),mpre(i+1));
  end
  i=find(mrec(2:end)~=mrec(1:end-1))+1;
  ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
  '''
  mrec = [0.0] + recall + [1.0]
  mpre = [0.0] + precision + [0.0]
  for i in range(len(mpre)-2, -1,-1):
    mpre[i] = max(mpre[i],mpre[i+1])
  idx = [i+1 for i in range(0,len(mrec)-1) if mrec[i+1] != mrec[i]]
  AP2012 = 0.0
  for i in idx:
    AP2012 +=( mrec[i]-mrec[i-1])*mpre[i]

  '''
  PASCAL VOC 2007 devkit
  ap=0;
  for t=0:0.1:1
      p=max(prec(rec>=t));
      if isempty(p)
          p=0;
      end
      ap=ap+p/11;
  end
  '''
  AP2007 = 0.0
  for t in range(0,11,1):
    idx = [i for i in range(0,len(recall)) if recall[i] >= float(t)/10.0]
    if len(idx) != 0:
      p = max( [precision[i] for i in idx] )
    else:
      p = 0.0
    AP2007 += p/11

  print 'AP2012:',AP2012
  print 'AP2007:',AP2007
  print 'MAX RECALL:',max(recall)
  output.write('AP='+str(AP2007))
  output.close()

def computePrecAt(tp,K):
  import numpy as np
  print 'Prec@K',
  for k in K:
    print '(',str(k),':',np.sum(tp[0:k])/float(k),')',
  print ''

def evaluateTopKDetectionsVsRecall(detectionsData, indexData, minOverlap, output, topks):
  numPositives = float( len(indexData) )
  topkRecall = []
  for topk in topks:
    groundTruth = loadGroundTruthAnnotations(indexData)
    detections = loadMaxDetectionsPerImage(detectionsData, topk)
    results = evaluateDetections(groundTruth, detections, minOverlap, output, overlapMeasure=ldet.overlap)
    recall = sum(results['tp']) / numPositives
    print topk, recall
    topkRecall.append([topk,recall])
  out = open(output,'w')
  for r in topkRecall:
    out.write(str(r[0]) + ' ' + str(r[1]) + '\n')
  return

def evaluateOverlapVsRecall(detectionsData, indexData, topK, output, overlaps):
  numPositives = float( len(indexData) )
  overlapRecall = []
  for ov in overlaps:
    groundTruth = loadGroundTruthAnnotations(indexData)
    detections = loadMaxDetectionsPerImage(detectionsData, topK)
    results = evaluateDetections(groundTruth, detections, ov, output, overlapMeasure=ldet.overlap)
    recall = sum(results['tp']) / numPositives
    print ov, recall
    overlapRecall.append([ov,recall])
  out = open(output,'w')
  for r in overlapRecall:
    out.write(str(r[0]) + ' ' + str(r[1]) + '\n')
  return

# Main Program
if __name__ == "__main__":
  params = cu.loadParams("overlap groundTruth detections output")
  indexData = [x.split() for x in open(params['groundTruth'])]
  detectionsData = [x.split() for x in open(params['detections'])]

  groundTruth = loadGroundTruthAnnotations(indexData)
  numPositives = len(indexData)
  print 'Annotated images:',len(groundTruth)
  print 'Ground Truth Bounding Boxes:',len(indexData)

  if params['overlap'].startswith('TOPK'):
    minOverlap = float(params['overlap'].replace('TOPK',''))
    evaluateTopKDetectionsVsRecall(detectionsData, indexData, minOverlap, params['output'], [1]+range(5,101,5))
    sys.exit()
  elif params['overlap'].startswith('ROV'):
    topK = int(params['overlap'].replace('ROV',''))
    evaluateOverlapVsRecall(detectionsData, indexData, topK, params['output'], [x/20.0 for x in range(1,21)])
    sys.exit()
  elif params['overlap'].startswith('OV'):
    overlapMeasure = ldet.overlap
    minOverlap = float(params['overlap'].replace('OV',''))
  else:
    overlapMeasure = ldet.IoU
    minOverlap = float(params['overlap'])
  
  detections = loadDetections(detectionsData)
  results = evaluateDetections(groundTruth,detections,minOverlap,params['output'],overlapMeasure,allowDuplicates=False)
  computePrecisionRecall(numPositives,results['tp'],results['fp'],params['output'])
  computePrecAt(results['tp'],[20,50,100,200,300,400,500])
  

