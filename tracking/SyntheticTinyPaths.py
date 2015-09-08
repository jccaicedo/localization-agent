import os,sys
import numpy as np
import pylab as pl
from numpy import random as rn

class RandomMovingTarget():

  def __init__(self, frameWidth, frameHeight, step):
    self.width = frameWidth
    self.height = frameHeight
    self.x = rn.randint(0, self.width)
    self.y = rn.randint(0, self.height)
    self.steps = [(0,-step,1,'Down'),(0,step,2,'Up'),(-step,0,3,'Left'),(step,0,4,'Right')]

  def move(self):
    rn.shuffle(self.steps)
    xn = self.x + self.steps[0][1]
    yn = self.y + self.steps[0][0]
    if xn >= 0 and xn < self.width and xn != self.x:
      self.x = xn
    elif yn >= 0 and yn < self.height and yn != self.y:
      self.y = yn
    else:
      return 5
    return self.steps[0][2]

def showSequence(seq,moves,channels=1):
  steps = {5:'Stay',1:'Down',2:'Up',3:'Left',4:'Right'}
  img = None
  for j in range(seq.shape[0]):
    if channels == 1:
      im = seq[j,:,:]
    elif channels == 2:
      im = seq[j,:,:,:]
      im = np.swapaxes(im, 0, 1)
      im = np.swapaxes(im, 1, 2)
      im = np.concatenate( (im, np.zeros((im.shape[0], im.shape[1], 1))), 2 )
    if img is None:
      img = pl.imshow(im)
    else:
      img.set_data(im)
    pl.title(steps[moves[j]])
    pl.draw()
    pl.pause(1.0/1)
  pl.close()

def generateDiffSeq(frames, width, height, patch):
  S = np.random.random( (frames, width, height) )
  S[S>=0.99] = 1
  S[S<0.99] = -1
  target = RandomMovingTarget(width-patch, height-patch, 4)
  moves = [5]
  for i in range(frames):
    S[i,target.x:target.x+patch, target.y:target.y+patch] = 1
    moves.append( target.move() )
  for i in range(frames-1,0,-1):
    S[i,:,:] = (S[i,:,:] - S[i-1,:,:])/2
  S[0,:,:] = 0; S[0,0,0] = 1; S[0,1,1] = -1;
  return S,moves

def generateMaskedSeq(frames, width, height, patch):
  S = np.random.random( (frames, 2, width, height) )
  S[S>=0.99] = 1
  S[S<0.99] = -1
  S[:,1,:,:] = 0
  target = RandomMovingTarget(width-patch, height-patch, 4)
  moves = [5]
  for i in range(frames):
    S[i, 0, target.x:target.x+patch, target.y:target.y+patch] = 1
    if i == 0:
      S[i, 1, target.x:target.x+patch, target.y:target.y+patch] = -1
    if i < frames-1:
      S[i+1, 1, target.x:target.x+patch, target.y:target.y+patch] = 1
    moves.append( target.move() )
  return S,moves

def generateDiagonalFrames(frames, width, height):
  S = np.random.random( (frames, width, height) )
  S[S>=0.99] = 1
  S[S<0.99] = -1
  target = np.random.random( (frames) )
  target[target >0.5] = 1 
  target[target<=0.5] = 2
  for i in range(frames):
    x = range(width)
    y = range(height)
    if target[i] == 2: x.reverse()
    S[i,x,y] = 1
  return S,target

