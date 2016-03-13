import theano as Theano
import numpy as NP
import theano.tensor as Tensor
import Tester

e = 5 # delta in pixels

S = NP.asarray([0.,0.,0.,0.], dtype=NP.float32)  # STAY 0
L = NP.asarray([-e,0.,-e,0.], dtype=NP.float32)  # LEFT 1
R = NP.asarray([ e,0., e,0.], dtype=NP.float32)  # RIGHT 2
U = NP.asarray([0., e,0., e], dtype=NP.float32)  # UP 3
D = NP.asarray([0.,-e,0.,-e], dtype=NP.float32)  # DOWN 4
I = NP.asarray([ e,-e,-e, e], dtype=NP.float32)  # ZOOM IN 5
O = NP.asarray([-e, e, e,-e], dtype=NP.float32)  # ZOOM OUT 6
F = NP.asarray([-e,0., e,0.], dtype=NP.float32)  # FATTER 7
T = NP.asarray([0.,-e,0., e], dtype=NP.float32)  # TALLER 8

ALL_ACTIONS = [S,L,R,U,D,I,O,F,T]

# The following function runs with Numpy in the CPU
def prepareTargets(boxes, imgSize):
    # Assume boxes for a batch of sequences (batch, time, coords)
    batchSize, seqLength, coords = boxes.shape
    iou = NP.zeros((batchSize, len(ALL_ACTIONS)))
    # Start with given box at time zero. Note that the action at time zero is always zero (Stay)
    targets = NP.zeros((batchSize, seqLength))
    bestIoU = NP.zeros((batchSize, seqLength))
    prev = boxes[:,0,:].copy()
    for t in range(1,seqLength):
        # Test all moves and compute IoU with ground truths
        for a in range(len(ALL_ACTIONS)):
          action = ALL_ACTIONS[a][NP.newaxis, :]
          moved = prev + action
          iou[:,a] = Tester.getIntOverUnion(boxes[:,t,:], moved)
        # Find and apply the best move to boxes in the current time
        targets[:,t] = NP.argmax(iou,axis=1)
        bestIoU[:,t] = NP.max(iou,axis=1)
        #print " # before",t,prev[0]
        for a in range(len(ALL_ACTIONS)):
            T = targets[:,t] == a
            prev += ALL_ACTIONS[a][NP.newaxis,:] * T[:, NP.newaxis]
        # Clip boxes
        prev[prev < 0.] = 0.
        prev[prev > imgSize] = imgSize-1
        #print " # after",t,prev[0],targets[0,t]
        if bestIoU[0,1] < 0.5:
            print ' == original boxes',boxes[0,0],boxes[0,1]
            print ' == transformed box',prev[0]
            print ' == bestIoU',bestIoU[0,1],'action',targets[0,1]
            import sys; sys.exit()
    print ' * target',targets[0,:]
    print ' * bestIoU',bestIoU[0,:]
    return targets

## Theano tensor operators
tS = Tensor.as_tensor_variable(S)  # STAY
tL = Tensor.as_tensor_variable(L)  # LEFT
tR = Tensor.as_tensor_variable(R)  # RIGHT
tU = Tensor.as_tensor_variable(U)  # UP
tD = Tensor.as_tensor_variable(D)  # DOWN
tI = Tensor.as_tensor_variable(I)  # ZOOM IN
tO = Tensor.as_tensor_variable(O)  # ZOOM OUT
tF = Tensor.as_tensor_variable(F)  # FATTER
tT = Tensor.as_tensor_variable(T)  # TALLER

ALL_T_ACTIONS = [tS,tL,tR,tU,tD,tI,tO,tF,tT]

# This function runs with Theano in the GPU
def moveBoxes(boxes, actionProbs, imgSize):
    actions = Tensor.argmax(actionProbs, axis=1)
    B = boxes.copy()
    for a in range(len(ALL_T_ACTIONS)):
        B = B + ALL_T_ACTIONS[a].dimshuffle('x',0) * Tensor.eq(actions,a).dimshuffle(0,'x')
    B -= Tensor.lt(B, 0.) * B
    B -= Tensor.gt(B, imgSize) * (B-imgSize)
    return B
