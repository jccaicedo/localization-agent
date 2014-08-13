# Training
modelsOutput=/home/caicedo/data/rcnn/models/
# Test
trainingList=/home/caicedo/data/pascal07/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt
#trainingList=/home/caicedo/data/rcnn/debug/aeroplaneDebugTrainval.txt
testList=/home/caicedo/data/pascal07/VOCdevkit/VOC2007/ImageSets/Main/test.txt
#testList=/home/caicedo/data/rcnn/debug/aeroplaneDebugTest.txt
#unlabeledList=/home/caicedo/data/rcnn/debug/aeroplane_extras.txt
unlabeledList=/home/caicedo/data/rcnn/lists/2012/trainvalSet2012.txt
featuresExt=fc6_neuron_cudanet_out
NMSThresholdTest=0.2
scoreThreshold="-10.0"
resultsOutput=/home/caicedo/data/rcnn/results/
# Evaluation
boxExpansion="10"
cost=0.001
expName=ASSL
overlap=0.5
components=4
hardMiningIters=2
topK=100
modelType='linear' # latent | linear | sgd
#for category in bicycle bird boat bottle bus car cat chair cow diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor; do
category=$1
posFeatures="/home/caicedo/data/rcnn/GPUCategoriesPascal07/"$category"0"
featuresDir="/home/caicedo/data/rcnn/GPUFeatures10Pascal0712/"
unlabeledGroundTruth="/home/caicedo/data/rcnn/lists/2012/trainval/"$category"_gt_boxes.txt"

function evaluate(){
  echo " ==> 0.8 OVERLAP"; 
  python evaluation.py 0.8 $1 $2 $3".8.result" 
  echo " ==> 0.5 OVERLAP"; 
  python evaluation.py 0.5 $1 $2 $3".5.result" 
}

## Initialize Detector on Training on Labeled Set
labeledModelFile=$modelsOutput"/"$category"_"$boxExpansion"_"$cost"_"$expName"_"$overlap"_labeled_0.txt"
time python trainDetector.py $posFeatures"."$featuresExt $trainingList $featuresDir $labeledModelFile $cost $overlap $hardMiningIters

for((iter=0;iter<3;iter++)); do

  ## Detections in Unlabeled Set
  unlabeledResultsFile=$resultsOutput"/"$category"_"$boxExpansion"_"$cost"_"$expName"_"$overlap"_"$hardMiningIters"_unlabeled_"$iter".out"
  time python detector.py $labeledModelFile"."$hardMiningIters $modelType $unlabeledList $featuresDir $featuresExt $NMSThresholdTest $scoreThreshold $unlabeledResultsFile
  evaluate $unlabeledGroundTruth $unlabeledResultsFile $unlabeledResultsFile

  ## Training on Unlabeled Set
  unlabeledModelFile=$modelsOutput"/"$category"_"$boxExpansion"_"$cost"_"$expName"_"$overlap"_unlabeled_"$iter".txt"
  time python trainOnTopRanked.py $unlabeledResultsFile NoLog $topK $unlabeledList $featuresDir $featuresExt $unlabeledModelFile $cost $overlap $hardMiningIters

  ## Detections in Labeled Set
  labeledResultsFile=$resultsOutput"/"$category"_"$boxExpansion"_"$cost"_"$expName"_"$overlap"_"$hardMiningIters"_labeled_"$iter".out"
  time python detector.py $unlabeledModelFile"."$hardMiningIters $modelType $trainingList $featuresDir $featuresExt $NMSThresholdTest $scoreThreshold $labeledResultsFile
  evaluate $posFeatures".idx" $labeledResultsFile $labeledResultsFile

  ## Training on Labeled Set with High Overlaping Detections
  nextIter=$((iter+1))
  labeledModelFile=$modelsOutput"/"$category"_"$boxExpansion"_"$cost"_"$expName"_"$overlap"_labeled_"$nextIter".txt"
  time python trainOnTopRanked.py $posFeatures"."$featuresExt $labeledResultsFile".result.log" $topK $trainingList $featuresDir $featuresExt $labeledModelFile $cost $overlap $hardMiningIters
 
done

## Last Evaluation on Unlabeled Set
unlabeledResultsFile=$resultsOutput"/"$category"_"$boxExpansion"_"$cost"_"$expName"_"$overlap"_"$hardMiningIters"_unlabeled_"$nextIter".out"
time python detector.py $labeledModelFile"."$hardMiningIters $modelType $unlabeledList $featuresDir $featuresExt $NMSThresholdTest $scoreThreshold $unlabeledResultsFile
evaluate $unlabeledGroundTruth $unlabeledResultsFile $unlabeledResultsFile


