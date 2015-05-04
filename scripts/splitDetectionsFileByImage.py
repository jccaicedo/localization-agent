import os,sys
import utils as cu

if __name__ == "__main__":
  params = cu.loadParams("detectionsFile outputDir")
  f = open(params['detectionsFile'])
  line = f.readline()
  img = ''
  imgOut = open(params['outputDir'] + '/tmp.region_rank','w')
  while line != '':
    parts = line.split()
    if parts[0] != img:
      imgOut.close()
      imgOut = open(params['outputDir'] + '/' + parts[0] + '.region_rank','w')
      img = parts[0]
    imgOut.write(line)
    line = f.readline()
  imgOut.close()
  f.close()
