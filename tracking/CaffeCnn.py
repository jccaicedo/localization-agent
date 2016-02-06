import caffe
import numpy as NP

class CaffeCnn(object):
    
    net = None
    transformer = None
    seqLength = None
    
    def __init__(self, imgHeight, imgWidth, deployPath, modelPath, caffeRoot, batchSize, seqLength, meanImage, layerKey):
        self.seqLength = seqLength
        self.batchSize = batchSize
        self.layerKey = layerKey
        self.net, self.transformer = self.setup(imgHeight, imgWidth, deployPath, modelPath, caffeRoot, meanImage)
    
    
    def forward(self, data):
        data = data.reshape(-1, data.shape[-3], data.shape[-2], data.shape[-1])
        self.net.blobs['data'].data[...] = NP.array([self.transformer.preprocess('data', image) for image in data])
        #TODO: which method is fastest: out or direct reference of layer?
        feats = self.net.forward(blobs=[self.layerKey])[self.layerKey]
        feats = feats.reshape(self.batchSize, self.seqLength, feats.shape[-3], feats.shape[-2], feats.shape[-1])
        
        return feats
    
    
    def setup(self, imgHeight, imgWidth, deployPath, modelPath, caffeRoot, meanImage):
        caffe.set_mode_gpu()
        net = caffe.Net(deployPath, modelPath, caffe.TEST)
    
        # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1))
        transformer.set_mean('data', NP.load(caffeRoot + meanImage).mean(1).mean(1)) # mean pixel
        transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
        transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
    
        net.blobs['data'].reshape(self.batchSize * self.seqLength, 3, imgHeight, imgWidth)
    
        return net, transformer
        
    def outputShape(self):
        return self.net.blobs[self.layerKey].data.shape
