import os,sys

import json
from PIL import Image

import utils as cu

def processEpisode(filename):
  data = json.load(open(filename))
  counts = {}
  stepsWithoutTrigger = 0
  for i in range(len(data['actions'])):
    if data['actions'][i] == 8 and data['rewards'][i] > 0:
      try: counts[stepsWithoutTrigger] += 1
      except: counts[stepsWithoutTrigger] = 1
      stepsWithoutTrigger = 0
    else:
      stepsWithoutTrigger += 1
  return counts

def getCount(c,i):
  try: return c[i]
  except: return 0

def testMemStats(memDir):
  files = os.listdir(memDir)
  stats = {}
  for f in files:
    if f.endswith('.txt'):
      counts = processEpisode(memDir + '/' + f)
      detections = sum([getCount(counts,i) for i in range(10)])
      if detections > 0:
        createSequence(memDir + '/' + f, '/home/jccaicedo/data/pascalImgs/' + f.replace('.txt','.jpg'), '/home/jccaicedo/data/hdSeq/' + f.replace('.txt','.png'))
      for k in counts.keys():
        try: stats[k] += counts[k]
        except: stats[k] = counts[k]
  return stats

def datasetStats(expDir):
  subdirs = ['A','A','B','B','C','C','D','D','E','E','F','F','G','G','H','H','I','I','J','J']
  categories = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
  stats = {}
  for i in range(len(subdirs)):
    mem = testMemStats(expDir + '/' + subdirs[i] + '/' + categories[i] + '/testMem/')
    print categories[i],mean(mem)
    for k in mem.keys():
      try: stats[k] += mem[k]
      except: stats[k] = mem[k]
  return stats

def mean(hist):
  total = sum(hist.values())
  area = 0
  for k in hist.keys():
    area += hist[k]*(k+0)
  return float(area)/float(total)

def createSequence(memFile, imageName, output):
  seq = Image.new("RGB", (1480, 200), "white")
  data = json.load(open(memFile))
  im = Image.open(imageName)
  regSize = 128
  margin = 20
  x,y = 10,10
  icons = readIcons()
  for i in range(10):
    crp = im.crop(map(int,data['boxes'][i]))
    region = crp.resize((regSize,regSize),Image.ANTIALIAS)
    seq.paste(region, (x,y))
    seq.paste(icons[data['actions'][i]],(x+32,y+128+1))

    x += regSize + margin
    if data['actions'][i] == 8:
      mark = getInhibitionMark(data['boxes'][i])
      im.paste((0,0,0), mark[0])
      im.paste((0,0,0), mark[1])
  seq.save(output)

def readIcons():
  path = '/home/jccaicedo/data/icons/'
  icons = []
  files = os.listdir(path)
  files.sort()
  for f in files:
    icons.append( Image.open(path + '/' + f) )
  for i in range(len(icons)):
    icons[i] = icons[i].resize((64,64), Image.ANTIALIAS)
  return icons

def getInhibitionMark(box):
  MARK_WIDTH = 0.1
  w = box[2]-box[0]
  h = box[3]-box[1]
  b1 = map(int, [box[0] + w*0.5 - w*MARK_WIDTH, box[1], box[0] + w*0.5 + w*MARK_WIDTH, box[3]])
  b2 = map(int, [box[0], box[1] + h*0.5 - h*MARK_WIDTH, box[2], box[1] + h*0.5 + h*MARK_WIDTH])
  return [b1, b2] 

def evaluateNumberOfDetections():
  subdirs = ['A','A','B','B','C','C','D','D','E','E','F','F','G','G','H','H','I','I','J','J']
  categories = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
  dist = {}
  all = {}
  logPath = '/home/jccaicedo/data/evalPascal07/exp01/XXX/YYY/results/YYY.out.scores.detections.log'
  gtPath = '/home/jccaicedo/data/lists/2007/test/YYY_test_bboxes.txt'

  for i in range(len(subdirs)):
    logFile = logPath.replace('XXX',subdirs[i]).replace('YYY',categories[i])
    gtFile = gtPath.replace('YYY',categories[i])

    data = [x.split() for x in open(logFile)]
    counts = {}
    for k in data:
      try: counts[k[0]] += int(k[-1])
      except: counts[k[0]] = int(k[-1])
    gt = cu.loadBoxIndexFile(gtFile)
    gtCounts = dict([ (x,len(gt[x])) for x in gt.keys()])
    for k in gtCounts.keys():
      objs = gtCounts[k]
      try: det = counts[k]
      except: det = 0
      missed = objs - det
      try: 
        dist[objs] += missed
        all[objs] += 1
      except: 
        dist[objs] = missed
        all[objs] = 1
  return dist,all
