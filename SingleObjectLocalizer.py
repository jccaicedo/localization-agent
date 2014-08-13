__author__ = "Juan C. Caicedo, caicedo@illinois.edu"

class SingleObjectLocalizer():

  adjustingPercent = 0.1
  
  # Actions
  ACCEPT         = 0
  REJECT         = 1
  EXPAND_TOP     = 2
  EXPAND_BOTTOM  = 3
  EXPAND_LEFT    = 4
  EXPAND_RIGHT   = 5
  REDUCE_TOP     = 6
  REDUCE_BOTTOM  = 7
  REDUCE_LEFT    = 8
  REDUCE_RIGHT   = 9
  
  def __init__(self, imgSize, initialBox):
    self.prevBox = initialBox
    self.nextBox = initialBox
    self.imgWidth = imgSize[0]
    self.imgHeight = imgSize[1]
    self.lastAction = float("inf")
    self.history = []

  def performAction(self, action):
    #print 'SingleObjectLocalizer::performAction(',action,')'
    ## Terminal States
    if self.lastAction == self.ACCEPT or self.lastAction == self.REJECT:
      return self.lastAction
    ## Adjust the current box
    elif action == self.EXPAND_TOP:    self.adjustCurrentBox(-1, 'y', 1)
    elif action == self.EXPAND_BOTTOM: self.adjustCurrentBox( 1, 'y', 3)
    elif action == self.EXPAND_LEFT:   self.adjustCurrentBox(-1, 'x', 0)
    elif action == self.EXPAND_RIGHT:  self.adjustCurrentBox( 1, 'x', 2)
    elif action == self.REDUCE_TOP:    self.adjustCurrentBox( 1, 'y', 1)
    elif action == self.REDUCE_BOTTOM: self.adjustCurrentBox(-1, 'y', 3)
    elif action == self.REDUCE_LEFT:   self.adjustCurrentBox( 1, 'x', 0)
    elif action == self.REDUCE_RIGHT:  self.adjustCurrentBox(-1, 'x', 2)
    elif action == self.ACCEPT or action == self.REJECT:
      pass
    else:
      print 'Unknown action',action
      return
    self.lastAction = action
    self.history.append(action)
    return self.lastAction
      
  def adjustCurrentBox(self, direction, axis, coordinate):
    ''' direction  : -1 or 1
        axis       : x or y
        coordinate : 0,1,2,3
    '''
    if axis == 'x':
      delta = (self.nextBox[2] - self.nextBox[0]) * self.adjustingPercent
      limit = self.imgWidth
    else:
      delta = (self.nextBox[3] - self.nextBox[1]) * self.adjustingPercent
      limit = self.imgHeight

    self.prevBox = [x for x in self.nextBox]
    self.nextBox[coordinate] += direction*delta

    if self.nextBox[coordinate] < 0 and self.nextBox[coordinate] < -delta:
      self.nextBox[coordinate] = -delta
    if self.nextBox[coordinate] > limit and self.nextBox[coordinate] - limit > delta:
      self.nextBox[coordinate] = limit + delta

