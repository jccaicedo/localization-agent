# Training
modelsOutput=/home/caicedo/data/cnnPatches/models/
# Test
trainingList=/home/caicedo/data/pascal07/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt
#trainingList=/home/caicedo/data/rcnn/debug/aeroplaneDebugTrainval.txt
testList=/home/caicedo/data/pascal07/VOCdevkit/VOC2007/ImageSets/Main/test.txt
#testList=/home/caicedo/data/rcnn/debug/aeroplaneDebugTest.txt
featuresExt=fc6_neuron_cudanet_out
NMSThresholdTest=0.5
scoreThreshold="-10.0"
resultsOutput=/home/caicedo/data/cnnPatches/results/
# Evaluation

cost=0.001
expName=9ScalePatchesNoDups
overlap=0.5 # Given an arbitrary window, how much it is allowed to overlap with a ground truth box to still be considered negative
modelType='linear' # latent | linear
for category in aeroplane bicycle bird boat bottle bus car cat chair cow diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor; do
#for category in aeroplane; do
  posFeatures="/home/caicedo/data/cnnPatches/SquarePatchesCategories2007/"$category"."$featuresExt
  featuresDir="/home/caicedo/data/cnnPatches/SquarePatchesFeatures2007/"
  #testGroundTruth="/home/caicedo/data/cnnPatches/testSamples/"$category"/boxes_file.txt"
  testGroundTruth="/home/caicedo/data/cnnPatches/lists/2007/test/"$category"_test_bboxes.txt"

  modelArgs="C:"$cost"!maxIters:10!"

  iterations=2
  ## Training
  modelFile=$modelsOutput"/"$category"_"$cost"_"$expName"_"$overlap".txt"
  #time python trainDetector.py $modelType $modelArgs $posFeatures $trainingList $featuresDir $modelFile $overlap $iterations

  ## Detections in Test Set
  resultsFile=$resultsOutput"/"$category"_"$cost"_"$expName"_"$overlap"_"$iterations".out"
  #time python detector.py $modelType $modelFile"."$iterations $testList $featuresDir $featuresExt $NMSThresholdTest $scoreThreshold $resultsFile
  python evaluation.py OV0.5 $testGroundTruth $resultsFile $resultsFile".result"
  python evaluation.py TOPK0.5 $testGroundTruth $resultsFile $resultsFile".result_topk"
  python evaluation.py ROV50 $testGroundTruth $resultsFile $resultsFile".result_rov"

done
