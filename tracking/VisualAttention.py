import theano as Theano
import theano.tensor as Tensor
import numpy as NP

def createGaussianMask(imgSize):
    R = Tensor.arange(imgSize, dtype=Theano.config.floatX)
    eps = 1e-8
    def mask(img, label):
        box = (label + 1.) * (imgSize / 2.) # Uncenter and rescale labels
        cx = (box[:, 3] + box[:, 1]) / 2.
        cy = (box[:, 2] + box[:, 0]) / 2.
        sx = (box[:,3] - cx)*0.60
        sy = (box[:,2] - cy)*0.60
        FX = Tensor.exp(-(R - cx.dimshuffle(0, 'x')) ** 2 / 2. / (sx.dimshuffle(0, 'x') ** 2 + eps))
        FY = Tensor.exp(-(R - cy.dimshuffle(0, 'x')) ** 2 / 2. / (sy.dimshuffle(0, 'x') ** 2 + eps))
        mask = (FX.dimshuffle(0, 1, 'x') * FY.dimshuffle(0, 'x', 1))
        mask = mask + 0.05
        mask = mask - Tensor.gt(mask, 1.0)*(mask-1.0)
        return img * mask.dimshuffle(0,'x',1,2)
    return mask

def useNoMask():
    return lambda img, box: img

def createSquaredMask(imgSize):
    R = Tensor.arange(imgSize, dtype=Theano.config.floatX)
    eps = 1e-8
    return lambda img, box: img

def getSquaredMasks(data, labels, context, alpha):
    l = labels.copy()
    # Expand boxes with context
    l[:,0:1] -= context
    l[:,2:3] += context
    # Fix out of bounds boxes
    l[l < 0] = 0
    l[l > data.shape[2]] = data.shape[2]
    l = NP.array(l, NP.int)
    # Create masks assuming (batch, width, height, channels)
    masks = alpha * NP.ones(data.shape)
    print masks.shape, l.shape
    for i in range(masks.shape[0]):
        masks[i, l[i,1]:l[i,3], l[i,0]:l[i,2], ...] = 1
    return masks
