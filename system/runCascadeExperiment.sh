# Training
modelsOutput=/home/caicedo/data/rcnn/models/
# Test
#trainingList=/home/caicedo/data/pascal07/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt
trainingList=/home/caicedo/data/rcnn/debug/aeroplaneDebugTrainval.txt
trainingProposalsFile=/home/caicedo/data/rcnn/training_proposals.txt
#negativesList=/home/caicedo/data/rcnn/debug/aeroplaneDebugNegatives.txt
masksDir=/home/caicedo/data/rcnn/masks/
#testList=/home/caicedo/data/pascal07/VOCdevkit/VOC2007/ImageSets/Main/test.txt
testList=/home/caicedo/data/rcnn/debug/aeroplaneDebugTest.txt
testProposalsFile=/home/caicedo/data/rcnn/test_proposals.txt
featuresExt=conv3_cudanet_out
NMSThresholdTest=0.2
scoreThreshold="-10.0"
resultsOutput=/home/caicedo/data/rcnn/results/
# Evaluation

boxExpansion="10"
cost=0.001
expName=cascade1
overlap=0.5
components=1
#modelArgs="C:"$cost"!"
modelType='linear' # latent | linear | selfPaced
#for category in bicycle bird boat bottle bus car cat chair cow diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor; do
for category in aeroplane; do
  posList="/home/caicedo/data/rcnn/lists/2007/positives/"$category"_trainval.txt"
  featuresDir="/home/caicedo/data/rcnn/cascadeFeatures07/regionsLevel1/"
  testGroundTruth="/home/caicedo/data/pascal07/boxes/test/"$category"_test_bboxes.txt"
  trainingGroundTruth="/home/caicedo/data/rcnn/lists/2007/trainval/"$category"_gt_bboxes.txt"

  modelArgs="C:"$cost"!components:"$components"!maxIters:10!featuresExt:"$featuresExt"!"

  iterations=2
  ## Training
  modelFile=$modelsOutput"/"$category"_"$boxExpansion"_"$cost"_"$expName"_"$overlap".txt"
  #time python trainCascade.py $modelType $modelArgs $category $posList $trainingList $masksDir $featuresDir $featuresExt $modelFile $overlap $iterations

  ## Detections in Training Set
  resultsFile=$resultsOutput"/"$category"_"$boxExpansion"_"$cost"_"$expName"_"$overlap"_"$iterations"_training.out"
  resultsDir='/home/caicedo/data/rcnn/testMasks/'
  #time python maskDetector.py $modelFile"."$iterations $trainingList $trainingProposalsFile $featuresDir $featuresExt $scoreThreshold $resultsFile
  python evaluation.py 0.5 $trainingGroundTruth $resultsFile $resultsFile".result"

  ## Extract features in the selected set of regions
  #cat $resultsFile".result.log" | awk '{print $1" "$2" "$3" "$4" "$5}' > $resultsFile".result.regions"
  #python extractCNNFeaturesPerImage.py $resultsFile".result.regions" '/home/caicedo/data/allimgs/' '/home/caicedo/data/rcnn/cascadeFeatures07/regionsLevel2/'

  ## Detections in Test Set
  resultsFile=$resultsOutput"/"$category"_"$boxExpansion"_"$cost"_"$expName"_"$overlap"_"$iterations"_test.out"
  resultsDir='/home/caicedo/data/rcnn/testMasks/'
  #time python maskDetector.py $modelFile"."$iterations $testList $testProposalsFile $featuresDir $featuresExt $scoreThreshold $resultsFile
  #python evaluation.py OV0.7 $testGroundTruth $resultsFile $resultsFile".result"

  ## Extract features in the selected set of regions
  #cat $resultsFile".result.log" | awk '{print $1" "$2" "$3" "$4" "$5}' > $resultsFile".result.regions"
  #python extractCNNFeaturesPerImage.py $resultsFile".result.regions" '/home/caicedo/data/allimgs/' '/home/caicedo/data/rcnn/cascadeFeatures07/regionsLevel2/'
done
