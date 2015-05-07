import os,sys
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import utils.utils as cu

import learn.rl.RLConfig as config

params = cu.loadParams("config caffeLog rlLog outdir")
config.readConfiguration(params["config"])

fig, ax = plt.subplots(nrows=2, ncols=3)
fig.set_size_inches(18.5,10.5)

# Parse Caffe Log
loss = []
for l in open(params['caffeLog']):
  if l.find('loss =') != -1:
    loss.append( float(l.split()[-1]) )
i = np.argmax(loss)
loss[i] = np.average(loss)
ax[0,0].plot(range(len(loss)), loss)
ax[0,0].set_title('QNetwork Loss')

# Parse RL output
avgRewards = []
epochRewards = []
epochRecall = []
epochIoU = []
epochLandmarks = []
validationLandmarks = []
positives = dict([ (i,0) for i in range(config.geti('outputActions')) ])
negatives = dict([ (i,0) for i in range(config.geti('outputActions')) ])
for l in open(params['rlLog']):
  if l.find('Agent::MemoryRecord') != -1:
    parts = l.split()
    action = int(parts[7])
    reward = float(parts[9])
    if reward > 0:
      positives[action] += 1
    else:
      negatives[action] += 1
  elif l.find('reset') != -1:
    avgRewards.append( float(l.split()[-1]) )
  elif l.find('Epoch Recall') != -1:
    epochRecall.append( float(l.split()[-1]) )
    epochRewards.append( np.average(avgRewards) )
    avgRewards = []
  elif l.find('MaxIoU') != -1:
    epochIoU.append( float(l.split()[-1]) )
  elif l.find('Epoch Landmarks:') != -1:
    epochLandmarks.append( float(l.split()[-1]) )
  elif l.find('Validation Landmarks') != -1:
    validationLandmarks.append( float(l.split()[-1]) )

# Draw all plots
ax[1,0].plot(range(len(epochRewards)), epochRewards)
ax[1,0].set_title('Average Reward Per Epoch')

recall = np.zeros( (len(epochRecall),2) )
recall[:,0] = epochRecall
recall[:,1] = epochLandmarks
ax[0,1].plot(range(len(epochRecall)), recall)
ax[0,1].set_title('Recall per Epoch')

ax[1,1].plot(range(len(epochIoU)), epochIoU)
ax[1,1].set_title('MaxIoU per Epoch')

pos = positives.keys(); pos.sort()
neg = negatives.keys(); neg.sort()
ind = np.arange(len(pos)); width = 0.35
rp = ax[0,2].bar(ind        , [positives[k] for k in pos], width=0.35, color='g' )
rn = ax[0,2].bar(ind + width, [negatives[k] for k in neg], width=0.35, color='r' )
ax[0,2].set_title('Distribution of rewards per action')
ax[0,2].legend( (rp[0], rn[0]), ('Positive','Negative') )
ax[0,2].set_xticks(ind + width)
ax[0,2].set_xticklabels( ('XU','YU','SU','AU','XD','YD','SD','AD','LM','SR') )

ax[1,2].plot(range(len(validationLandmarks)), validationLandmarks)
ax[1,2].set_title('% Landmarks in Validation Set')


# Save Plot
plt.savefig(params['outdir'] + '/report' + config.get('category') + '.png')
