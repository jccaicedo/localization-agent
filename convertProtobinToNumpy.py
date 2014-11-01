from caffe import io as c
import numpy as np
import os,sys

if len(sys.argv) < 3:
  print 'Use: convertProtobinToNumpy protobinFile numpyOutput'
  sys.exit()

protoData = c.caffe_pb2.BlobProto()
f = open(sys.argv[1],'rb')
protoData.ParseFromString(f.read())
f.close()
array = c.blobproto_to_array(protoData)
np.save(sys.argv[2],array[0].swapaxes(1, 0).swapaxes(2,1)[:, :, ::-1])
A = np.load(sys.argv[2]+'.npy')
print 'Final matrix shape:',A.shape
