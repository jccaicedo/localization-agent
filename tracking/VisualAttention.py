import theano as Theano
import theano.tensor as Tensor

def createGaussianMask(imgSize):
    R = Tensor.arange(imgSize, dtype=Theano.config.floatX)
    eps = 1e-8
    def mask(img, label):
        box = (label + 1.) * (imgSize / 2.) # Uncenter and rescale labels
        cx = (box[:, 3] + box[:, 1]) / 2.
        cy = (box[:, 2] + box[:, 0]) / 2.
        sx = (box[:,3] - cx)*0.75
        sy = (box[:,2] - cy)*0.75
        FX = Tensor.exp(-(R - cx.dimshuffle(0, 'x')) ** 2 / 2. / (sx.dimshuffle(0, 'x') ** 2 + eps))
        FY = Tensor.exp(-(R - cy.dimshuffle(0, 'x')) ** 2 / 2. / (sy.dimshuffle(0, 'x') ** 2 + eps))
        mask = (FX.dimshuffle(0, 1, 'x') * FY.dimshuffle(0, 'x', 1))
        return img * mask.dimshuffle(0,'x',1,2)
    return mask

def useNoMask():
    return lambda img, box: img

def createSquaredMask(imgSize):
    R = Tensor.arange(imgSize, dtype=Theano.config.floatX)
    eps = 1e-8
    return lambda img, box: img
