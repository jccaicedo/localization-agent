import sys,os
import utils as cu

params = cu.loadParams('fullList positivesList output')

full = [x for x in open(params['fullList'])]
positives = [x for x in open(params['positivesList'])]
out = open(params['output'],'w')
for r in full:
  if r not in positives:
    out.write(r)
out.close()
