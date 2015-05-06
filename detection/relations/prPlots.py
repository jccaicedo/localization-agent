import os,sys
import matplotlib.pyplot as plt

def loadPrecRecData(filename):
  data = [map(float,x.split()) for x in open(filename)]
  ap = data.pop(-1)
  xd = [x[0] for x in data]
  yd = [y[1] for y in data]
  return xd, yd, ap

dir = '/home/caicedo/data/rcnn/regionSearchResults/fifth/'
categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
bigFile = '_big_0.001_region_search_fifth_0.7_2.out.result'
tightFile = '_tight_0.001_region_search_fifth_0.5_2.out.result'
insideFile = '_inside_0.001_region_search_fifth_0.7_2.out.result'

for k in categories:
  xbig, ybig, apbig = loadPrecRecData(dir + k + bigFile)
  xtig, ytig, aptig = loadPrecRecData(dir + k + tightFile)
  xins, yins, apins = loadPrecRecData(dir + k + insideFile)
  fig = plt.figure()
  fig.suptitle(k, fontsize=14, fontweight='bold')
  plt.plot(xbig, ybig, 'r-,', xtig, ytig, 'g-,', xins, yins, 'b-,')
  plt.legend(['Big    AP='+'{:5.4f}'.format(apbig[1]),'Tight  AP='+'{:5.4f}'.format(aptig[1]),'Inside AP='+'{:5.4f}'.format(apins[1])])
  #plt.show()
  plt.savefig(dir + k + '.png')
