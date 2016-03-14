import theano as Theano
import theano.tensor as Tensor
import numpy as NP
import cv2

CONTEXT = 4
ALPHA = 0.1

def normalLabels(labels, imgSize):
    return labels / (imgSize / 2.) - 1.

def unnormedLabels(labels, imgSize):
    return (labels + 1.) * (imgSize / 2.)

def centeredLabels(labels, imgSize):
    return labels - (imgSize / 2.)

def uncenteredLabels(labels, imgSize):
    return labels + (imgSize / 2.)

# TODO: Parameterize the use of the following functions
stdLabels = normalLabels # centeredLabels
stdBoxes = unnormedLabels # uncenteredLabels

def createGaussianMasker(imgSize):
    R = Tensor.arange(imgSize, dtype=Theano.config.floatX)
    eps = 1e-8
    alpha = ALPHA
    def mask(img, label):
        box = stdBoxes(label, imgSize)
        cx = (box[:, 3] + box[:, 1]) / 2.
        cy = (box[:, 2] + box[:, 0]) / 2.
        sx = (box[:,3] - cx)*0.60
        sy = (box[:,2] - cy)*0.60
        FX = Tensor.exp(-(R - cx.dimshuffle(0, 'x')) ** 2 / 2. / (sx.dimshuffle(0, 'x') ** 2 + eps))
        FY = Tensor.exp(-(R - cy.dimshuffle(0, 'x')) ** 2 / 2. / (sy.dimshuffle(0, 'x') ** 2 + eps))
        m = (FX.dimshuffle(0, 1, 'x') * FY.dimshuffle(0, 'x', 1))
        m = m + alpha
        m = m - Tensor.gt(m, 1.0) * (m - 1.0)
        return img * m.dimshuffle(0,'x',1,2)
    return mask

def useNoMask():
    return lambda img, box: img

def createSquareMasker(imgSize):
    R = Tensor.arange(imgSize, dtype=Theano.config.floatX)
    eps = 1e-8
    alpha = ALPHA
    def mask(img, label):
        box = stdBoxes(label, imgSize)
        FX = Tensor.gt(R, box[:,1].dimshuffle(0,'x')) * Tensor.le(R, box[:,3].dimshuffle(0,'x'))
        FY = Tensor.gt(R, box[:,0].dimshuffle(0,'x')) * Tensor.le(R, box[:,2].dimshuffle(0,'x'))
        m = (FX.dimshuffle(0, 1, 'x') * FY.dimshuffle(0, 'x', 1))
        m = m + alpha - Tensor.gt(m, 0.) * alpha
        return img * m.dimshuffle(0,'x',1,2)
    return mask

def createSquareChannelMasker(imgSize):
    R = Tensor.arange(imgSize, dtype=Theano.config.floatX)
    eps = 1e-8
    def mask(img, label):
        box = stdBoxes(label, imgSize)
        FX = Tensor.gt(R, box[:,1].dimshuffle(0,'x')) * Tensor.le(R, box[:,3].dimshuffle(0,'x'))
        FY = Tensor.gt(R, box[:,0].dimshuffle(0,'x')) * Tensor.le(R, box[:,2].dimshuffle(0,'x'))
        m = 2.*(FX.dimshuffle(0, 1, 'x') * FY.dimshuffle(0, 'x', 1)) - 1.
        return Tensor.set_subtensor(img[:,-1,:,:], m)
    return mask

# TODO: standardize the use of the term labels (for learning) and boxes (for actual usable coordinates)
def getSquaredMasks(data, labels):
    l = labels.copy()
    # Expand boxes with context
    l[:,0:1] -= CONTEXT
    l[:,2:3] += CONTEXT
    # Fix out of bounds boxes
    l[l < 0] = 0
    l[l > data.shape[2]] = data.shape[2]
    l = NP.array(l, NP.int)
    # Create masks assuming (batch, channels, width, height)
    masks = ALPHA * NP.ones(data.shape)
    for i in range(masks.shape[0]):
        masks[i, :, l[i,1]:l[i,3], l[i,0]:l[i,2]] = 1
    return masks

def getSquaredMaskChannel(data, labels):
    # Assuming input shape=(batch, channels, width, height)
    b,c,w,h = data.shape
    l = labels.copy()
    # Expand boxes with context
    l[:,0:1] -= CONTEXT
    l[:,2:3] += CONTEXT
    # Fix out of bounds boxes
    l[l < 0] = 0
    l[l > w] = w
    l = NP.array(l, NP.int)
    # Create masks
    masks = -1 * NP.ones((b,w,h))
    for i in range(masks.shape[0]):
        masks[i, l[i,1]:l[i,3], l[i,0]:l[i,2]] = 1.
    return masks

def buildAttention(useAttention, imgSize):
    if useAttention == 'gaussian':
        attention = createGaussianMasker(imgSize)
    elif useAttention == 'square':
        attention = createSquareMasker(imgSize)
    elif useAttention == 'squareChannel':
        attention = createSquareChannelMasker(imgSize)
    else:
        attention = useNoMask()
    return attention

def rgb2gray(rgb):
    return NP.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def framesFlow(f1, f2):
    frame1 = rgb2gray(f1)
    frame2 = rgb2gray(f2)
    f = cv2.calcOpticalFlowFarneback(frame1, frame2, 0.5, 1, 3, 15, 3, 1.2, 0)
    f[...,0] = cv2.normalize(f[...,0], None, 0, 255, cv2.NORM_MINMAX)
    f[...,1] = cv2.normalize(f[...,0], None, 0, 255, cv2.NORM_MINMAX)
    return f

def computeFlowFromBatch(data):
    # Assume tensor with (batch, frame, width, height, channel)
    b,f,w,h,c = data.shape
    flow = NP.zeros( (b,f,w,h,2) )
    for i in range(b): # Each batch
        for j in range(f-1): # Each frame
            flow[i,j+1,...] = framesFlow(data[i,j,...], data[i,j+1,...])
    return flow

def computeFlowFromList(data):
    f = len(data)
    w,h,c = data[0].shape
    flow = NP.zeros( (f,w,h,2) )
    for i in range(f-1):
        flow[i+1,...] = framesFlow(data[i],data[i+1])
    return flow
