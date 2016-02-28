import theano as Theano
import theano.tensor as Tensor
import numpy as NP

def createGaussianMasker(imgSize):
    R = Tensor.arange(imgSize, dtype=Theano.config.floatX)
    eps = 1e-8
    alpha = 0.1
    def mask(img, label):
        box = (label + 1.) * (imgSize / 2.) # Uncenter and rescale labels
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
    alpha = 0.1
    def mask(img, label):
        box = (label + 1.) * (imgSize / 2.) # Uncenter and rescale labels
        FX = Tensor.gt(R, box[:,1].dimshuffle(0,'x')) * Tensor.le(R, box[:,3].dimshuffle(0,'x'))
        FY = Tensor.gt(R, box[:,0].dimshuffle(0,'x')) * Tensor.le(R, box[:,2].dimshuffle(0,'x'))
        m = (FX.dimshuffle(0, 1, 'x') * FY.dimshuffle(0, 'x', 1))
        m = m + alpha - Tensor.gt(m, 0.) * alpha
        return img * m.dimshuffle(0,'x',1,2)
    return mask

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
    for i in range(masks.shape[0]):
        masks[i, l[i,1]:l[i,3], l[i,0]:l[i,2], ...] = 1
    return masks
