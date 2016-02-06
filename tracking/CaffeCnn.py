import caffe
import numpy as NP

class CaffeCnn(object):
    
    net = None
    transformer = None
    seqLength = None
    
    def __init__(self, imgHeigh, imgWidth, deployPath, modelPath, caffeRoot, batchSize, seqLength):
        self.seqLength = seqLength
        self.batchSize = batchSize
        self.net, self.transformer = self.setup(imgHeigh, imgWidth, deployPath, modelPath, caffeRoot)
    
    
    def forward(self, data):
        data = data.reshape(-1, data.shape[-3], data.shape[-2], data.shape[-1])
        self.net.blobs['data'].data[...] = NP.array([self.transformer.preprocess('data', image) for image in data])
        out = self.net.forward()
        feat = self.net.blobs['inception_5b/output'].data
        feat = feat.reshape(self.batchSize, self.seqLength, feat.shape[-3], feat.shape[-2], feat.shape[-1])
        
        return out, feat
    
    
    def setup(self, imgHeigh, imgWidth, deployPath, modelPath, caffeRoot):
        caffe.set_mode_gpu()
        net = caffe.Net(deployPath, modelPath, caffe.TEST)
    
        # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1))
        transformer.set_mean('data', NP.load(caffeRoot + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
        transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
        transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
    
        net.blobs['data'].reshape(self.batchSize * self.seqLength, 3, imgHeigh, imgWidth)
    
        return net, transformer