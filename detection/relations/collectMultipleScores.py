import os, sys
import utils as cu
import regionSelection as rs
import libDetection as det

MINUS_INF = -99999

def readScoresFile(file):
  records = {}
  input = open(file)
  line = input.readline()
  while line != '':
    parts = line.split()
    try:
      records[ parts[0] ][' '.join(parts[2:6])] = parts[1] # bbox:score
    except:
      records[ parts[0] ] = {' '.join(parts[2:6]):parts[1]} # bbox:score
    line = input.readline()
  input.close()
  return records

def pickBest(bsc, tsc, isc):
  score,type = MINUS_INF,'?'
  if bsc > max(tsc,isc):
    score,type = bsc,'big'
  elif tsc > max(bsc,isc):
    score,type = tsc,'tight'
  elif isc > max(bsc,tsc):
    score,type = isc,'inside'
  return (score,type)

def mergeScores(big, tight, inside):
  records = {}
  for img in big.keys():
    records[img] = {}
    for box in big[img].keys():
      try: bsc = big[img][box] 
      except: bsc = MINUS_INF
      try: tsc = tight[img][box] 
      except: tsc = MINUS_INF
      try: isc = inside[img][box] 
      except: isc = MINUS_INF
      score,type = pickBest(bsc,tsc,isc)
      records[img][box] = {'score':score,'type':type,'box':map(float,box.split())}
  return records

def evaluateType(boxData,gt,threshold):
  b = rs.big(boxData['box'],gt)
  t = rs.tight(boxData['box'],gt)
  i = rs.inside(boxData['box'],gt)
  
  if boxData['type'] == 'big' and float(boxData['score']) >= threshold:
    if b:   result = {'big':'tp'}
    elif t: result = {'big':'fp', 'tight':'fn'}
    elif i: result = {'big':'fp', 'inside':'fn'}
    else:   result = {'big':'fp'}
  elif boxData['type'] == 'tight' and float(boxData['score']) >= threshold:
    if b:   result = {'tight':'fp', 'big':'fn'}
    elif t: result = {'tight':'tp'}
    elif i: result = {'tight':'fp', 'inside':'fn'}
    else:   result = {'tight':'fp'}
  elif boxData['type'] == 'inside' and float(boxData['score']) >= threshold:
    if b:   result = {'inside':'fp', 'big':'fn'}
    elif t: result = {'inside':'fp', 'tight':'fn'}
    elif i: result = {'inside':'tp'}
    else:   result = {'inside':'fp'}
  else:
    result = {boxData['type']:'tn'}
  
  return result
  

if __name__ == "__main__":
  params = cu.loadParams("bigFile tightFile insideFile groundTruthsFile threshold outputDir")

  big = readScoresFile(params['bigFile'])
  tight = readScoresFile(params['tightFile'])
  inside = readScoresFile(params['insideFile'])
  threshold = float(params['threshold'])
  results = mergeScores(big,tight,inside)

  groundTruths = cu.loadBoxIndexFile(params['groundTruthsFile'])

  counts = {'big': {'tp':0,'tn':0,'fp':0,'fn':0}, 'tight':{'tp':0,'tn':0,'fp':0,'fn':0}, 'inside':{'tp':0,'tn':0,'fp':0,'fn':0}}
  allBoxes = 0
  for img in results.keys():
    try:
      boxes = groundTruths[img]
      imageOK = True
    except:
      imageOK = False
    if imageOK:
      out = open(params['outputDir']+'/'+img+'.region_rank','w')
      for box in results[img].keys():
        allBoxes += 1
        maxIoU,assigned = 0,[0,0,0,0]
        for gt in groundTruths[img]:
          iou = det.IoU(results[img][box]['box'],gt)
          if iou > maxIoU: 
            assigned = gt
            maxIoU = iou
        res = evaluateType(results[img][box],assigned,threshold)
        for r in res.keys():
          counts[r][res[r]] += 1
        if res[ results[img][box]['type'] ] == 'tp':
          correct = '1'
        else:
          correct = '0'
        out.write( img + ' ' + results[img][box]['score'] + ' ' + box + ' ' + results[img][box]['type'] + ' ' + correct + '\n')
      out.close()
  
  print 'AllBoxes:',allBoxes
  for type in counts.keys():
    print type,':',
    for measure in counts[type]:
      print measure,counts[type][measure],
    print ''
