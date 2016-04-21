import theano as Theano
import theano.tensor as Tensor
import numpy as NP
import cv2
import lasagne

CONTEXT = 4
ALPHA = 0.1

def normalLabels(boxes, imgSize):
    return boxes / (imgSize / 2.) - 1.

def unnormedBoxes(labels, imgSize):
    return (labels + 1.) * (imgSize / 2.)

def centeredLabels(boxes, imgSize):
    return boxes - (imgSize / 2.)

def uncenteredBoxes(labels, imgSize):
    return labels + (imgSize / 2.)

def boxes2params(boxes, imgSize):
    # Params: Sx, Sy, tx, ty
    params = NP.zeros(boxes.shape)
    Acs = NP.array([[0,2,-1],[2,0,-1],[0,0,1]])
    for i in range(boxes.shape[0]):
        normPoints = boxes[i,...].reshape((boxes.shape[1], 2, 2)) / imgSize
        paddingOnes = NP.ones((boxes.shape[1], 2, 1))
        normBoxPoints = NP.concatenate([normPoints, paddingOnes], axis=2).swapaxes(2,1)
        Pc = NP.dot(Acs[NP.newaxis, ...], normBoxPoints).swapaxes(1,2)[0]
        Cc = (Pc[...,0] + Pc[...,1])[...,:-1]/2.
        Ss = (Pc[...,1] - Pc[...,0])[...,:-1]/2.
        params[i,...] = NP.stack([Ss[...,0], Ss[...,1], Cc[...,0], Cc[...,1]]).T
    return params

def params2boxes(params, imgSize):
    # Assuming equal width and height
    side = imgSize/2.
    H = params[...,0] * side
    W = params[...,1] * side
    Cy = (params[...,2] + 1) * side
    Cx = (params[...,3] + 1) * side
    boxes = NP.stack([Cx-W, Cy-H, Cx+W, Cy+H], axis=2)
    return boxes

# TODO: Parameterize the use of the following functions
stdLabels = boxes2params # centeredLabels
stdBoxes = params2boxes # uncenteredLabels

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

def prediction2params(prediction):
    Sx = prediction[:,0]
    Sy = prediction[:,1]
    tx = prediction[:,2]
    ty = prediction[:,3]
    z = Tensor.zeros_like(Sx)
    params = Tensor.stack([Sx, z, tx, z, Sy, ty], axis=0)
    return params.dimshuffle(1,0)

def params2prediction(params):
    Sx = params[:,0]
    Sy = params[:,4]
    tx = params[:,2]
    ty = params[:,5]
    prediction = Tensor.stack([Sx, Sy, tx, ty], axis=0)
    return prediction.dimshuffle(1,0)

def createSpatialTransformer(imgSize, channels):
    l_in = lasagne.layers.InputLayer((1,channels,imgSize,imgSize))
    l_loc_shape = (None, 6)
    l_trans = lasagne.layers.TransformerLayer(l_in, l_loc_shape, downsample_factor=1)
    def transform(imgs, params):
        #params_with_margin = Tensor.set_subtensor(params[:,0], params[:,0]+0.1)
        #params_with_margin = Tensor.set_subtensor(params_with_margin[:,4], params[:,4]+0.1)
        return l_trans.get_output_for((imgs, params))
    return transform

def multAffineTransforms(A, B):
    s = A.shape[0]
    Z = Tensor.zeros( (s, 2) )
    O = Tensor.ones( (s, 1) )
    Ap = Tensor.concatenate( (prediction2params(A), Z, O), axis=1 )
    Bp = Tensor.concatenate( (prediction2params(B), Z, O), axis=1 )
    Cp = Theano.scan(Tensor.dot, sequences=[Ap.reshape([s, 3, 3]), Bp.reshape([s, 3, 3])])
    R = Cp[0].reshape([s, 9])
    R = R[:,0:6]
    return R #prediction2params(A)*prediction2params(B)
    

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

def buildAttention(useAttention, imgSize, channels=3):
    if useAttention == 'gaussian':
        attention = createGaussianMasker(imgSize)
    elif useAttention == 'square':
        attention = createSquareMasker(imgSize)
    elif useAttention == 'squareChannel':
        attention = createSquareChannelMasker(imgSize)
    elif useAttention == 'spatialTransformer':
        attention = createSpatialTransformer(imgSize, channels)
    else:
        attention = useNoMask()
    return attention

###########################
## OPTIC FLOW COMPUTATIONS
###########################

def rgb2gray(rgb):
    return NP.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def framesFlow(f1, f2):
    frame1 = rgb2gray(f1)
    frame2 = rgb2gray(f2)
    #Farnebacks flow parameters
    pyr_scale = 0.5
    levels = 3 #1
    winsize = 15 #3
    iterations = 3 #15
    poly_n = 5 #3
    poly_sigma = 1.2 #1.2
    flags = 0 #0
    #Choose correct function invocation according to cv2 version
    if cv2.__version__.startswith('3'):
        f = cv2.calcOpticalFlowFarneback(frame1, frame2, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
    else:
        #TODO: Use created variables to ease interpretation and tweaking and check correct placement
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
