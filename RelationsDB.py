import os,sys
import utils as cu
import MemoryUsage
import numpy as np
import scipy.io
import random
from utils import tic, toc

class RelationsDB():

  def __init__(self, dbDir, randomize):
    self.images = [img for img in os.listdir(dbDir) if img.endswith('.mat')]
    self.dir = dbDir
    self.boxes = []
    self.scores = []
    self.image = ''
    self.idx = 0
    self.random = randomize
    if randomize:
      random.shuffle(self.images)

  def loadNext(self):
    if self.idx < len(self.images):
      data = scipy.io.loadmat(self.dir + '/' + self.images[self.idx])
      self.boxes = data['boxes']
      self.scores = data['scores']
      self.image = self.images[self.idx].replace('.mat','')
      self.idx += 1
    elif self.random:
      random.shuffle(self.images)
      self.idx = 0
      self.loadNext()
    else:
      self.boxes = []
      self.scores = []
      self.image = ''

class DBBuilder():

  def __init__(self, scoresDir, proposals):
    self.dir = scoresDir
    self.files = os.listdir(scoresDir)
    self.files.sort()
    self.categories = []
    for f in self.files:
      names = f.split('_')[0:2]
      cat = '_'.join(names)
      self.categories.append( cat )
    self.categories.sort()
    self.imgBoxes = {}
    self.scores = {}
    boxes = cu.loadBoxIndexFile(proposals)
    for img in boxes.keys():
      self.imgBoxes[img] = np.array(boxes[img])
      self.scores[img] = -10*np.ones( (len(boxes[img]), len(self.categories)) )

  def parseDir(self):
    for f in range(len(self.files)):
      self.readScoresFile(f)

  def readScoresFile(self, fileIdx):
    print 'Parsing',self.files[fileIdx]
    records = {}
    data = [x.split() for x in open(self.dir+'/'+self.files[fileIdx])]
    for d in data:
      key = d[0] + ' ' + ' '.join(d[2:6])
      records[key] = float(d[1])
    for img in self.imgBoxes.keys():
      for i in range(len(self.imgBoxes[img])):
        box = map( int, self.imgBoxes[img][i,:].tolist() )
        key = img + ' ' + ' '.join( map(str, box) )
        try: score = records[key]
        except: score = -10.0
        self.scores[img][i,fileIdx] = score

  def saveDB(self, outputDir):
    for img in self.imgBoxes.keys():
      data = {'boxes':self.imgBoxes[img], 'scores':self.scores[img]}
      scipy.io.savemat(outputDir+'/'+img+'.mat', data, do_compression=True)
    out = open(outputDir+'/categories.txt','w')
    for c in self.categories:
      out.write(c + '\n')
    out.close()

if __name__ == "__main__":
  params = cu.loadParams('scoresDirectory proposalsFile outputDir')
  cu.mem('Program started')
  lap = tic()
  builder = DBBuilder(params['scoresDirectory'], params['proposalsFile'])
  lap = toc('Proposals loaded', lap)
  cu.mem('DB initialized')
  builder.parseDir()
  lap = toc('Directory parsed', lap)
  cu.mem('All files read')
  builder.saveDB(params['outputDir'])
  lap = toc('Database saved', lap)
  cu.mem('Program ends')

