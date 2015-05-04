# Training
modelsOutput=/home/caicedo/data/rcnnPascal/models/
# Test
trainingList=/home/caicedo/data/cnnPatches/lists/2007/trainvalSet2007.txt
testList=/home/caicedo/data/cnnPatches/lists/2007/testSet2007.txt
featuresExt=fc7
NMSThresholdTest=0.3
scoreThreshold="-10.0"
resultsOutput=/home/caicedo/data/rcnnPascal/results/
# Evaluation

cost=0.001
expName=RCNN
overlap=0.3 # Given an arbitrary window, how much it is allowed to overlap with a ground truth box to still be considered negative
modelType='linear' # latent | linear
#for category in aeroplane bicycle bird boat bottle bus car cat chair cow diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor; do
for category in aeroplane; do
  posFeatures="/home/caicedo/data/relationsRCNN/python/features/tight/"$category"."$featuresExt
  featuresDir="/home/caicedo/data/rcnnPascal/FineTunedFeatures07/"
  testGroundTruth="/home/caicedo/data/cnnPatches/lists/2007/test/"$category"_test_bboxes.txt"

  modelArgs="C:"$cost"!maxIters:10!"

  iterations=4
  ## Training
  modelFile=$modelsOutput"/"$category"_"$cost"_"$expName"_"$overlap".txt"
  #time python trainDetector.py $modelType $modelArgs $posFeatures $trainingList $featuresDir $modelFile $overlap $iterations

  ## Detections in Test Set
  resultsFile=$resultsOutput"/"$category"_"$cost"_"$expName"_"$overlap"_"$iterations".out"
  #time python detector.py $modelType $modelFile"."$iterations $testList $featuresDir $featuresExt $NMSThresholdTest $scoreThreshold $resultsFile
  python evaluation.py 0.5 $testGroundTruth $resultsFile $resultsFile".result"

  #python evaluation.py OV0.5 $testGroundTruth $resultsFile $resultsFile".result"
  #python evaluation.py TOPK0.5 $testGroundTruth $resultsFile $resultsFile".result_topk"
  #python evaluation.py ROV50 $testGroundTruth $resultsFile $resultsFile".result_rov"

done
