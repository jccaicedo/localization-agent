import os,sys

# PATHS TO DATA
imageDir = '/u/sciteam/caicedor/scratch/pascalImgs/'
#candidatesFile = '/u/sciteam/caicedor/cnnPatches/trainingDetections/aeroplane_0.001_9ScalePatchesNoDups_0.5_2.out.result.log'
#candidatesFile = '/u/sciteam/caicedor/cnnPatches/trainingDetections/allTrainingDetections2007.txt'
candidatesFile = '/u/sciteam/caicedor/cnnPatches/trainingDetections/allDetectionsNoPerson.txt'

#groundTruth = '/u/sciteam/caicedor/cnnPatches/lists/2007/trainval/aeroplane_gt_bboxes.txt'
groundTruth = '/u/sciteam/caicedor/cnnPatches/lists/2007/trainval/all_objects_no_person.dat'
networkDir = '/u/sciteam/caicedor/scratch/rldetector01/'

# EXPLORATION PARAMETERS
maximumEpochs = 1
numInteractions = 500000
MAX_CANDIDATES_PER_IMAGE = 10
MIN_POSITIVE_OVERLAP = 0.5 # COVERAGE MEASURE
MAX_MOVES_ALLOWED = 20

# NETWORK FILES
TOOLS = "/u/sciteam/caicedor/caffe/build/tools/"
SOLVER_FILE="solver.prototxt"
SNAPSHOOT_PREFIX="finetune_network_train" # Taken from the solver file
PRETRAINED_MODEL="caffe_imagenet_train_iter_450000"
MEAN_IMAGE_PICKLE = "/u/sciteam/caicedor/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy"

# TRAINING PARAMETERS
trainingIterationsPerBatch = 40
replayMemorySize = 500000
percentOfValidation = 0.05

# DEPLOY PARAMETERS
imageSize = 256
deployBatchSize = 10
contextPad = 16
cropSize = 227
outputActions = 10
