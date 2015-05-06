# Training
modelsOutput=/home/caicedo/data/rcnn/models/
# Test
#trainingList=/home/caicedo/data/pascal07/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt
trainingList=/home/caicedo/data/rcnn/debug/aeroplaneDebugTrainval.txt
#testList=/home/caicedo/data/pascal07/VOCdevkit/VOC2007/ImageSets/Main/test.txt
testList=/home/caicedo/data/rcnn/debug/aeroplaneDebugTest.txt
featuresExt=fc6_neuron_cudanet_out
NMSThresholdTest=0.2
scoreThreshold="-10.0"
resultsOutput=/home/caicedo/data/rcnn/results/
# Evaluation

cost=0.001
expName=subCategories
overlap=0.3
hardNegativeIters=2
totalSubCategories=4
modelArgs="C:"$cost"!"
for category in bicycle bird boat bottle bus car cat chair cow diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor; do
#for category in aeroplane; do
  posFeatures="/home/caicedo/data/rcnn/GPUCategoriesPascal07/"$category"0."$featuresExt
  featuresDir="/home/caicedo/data/rcnn/GPUFeatures10Pascal07/"
  testGroundTruth="/home/caicedo/data/pascal07/boxes/test/"$category"_test_bboxes.txt"

  ## Training
  subcategory=0
  modelFile=$modelsOutput"/"$category"_"$cost"_"$expName"_"$overlap"_"$subcategory".txt"
  labelsFile=$modelsOutput"/"$category"_"$cost"_"$expName"_"$overlap"_"$totalSubCategories".txt.subcategories"
  time python subcatDetector.py $modelArgs $posFeatures $trainingList $featuresDir $modelFile $overlap $hardNegativeIters $labelsFile $totalSubCategories 0 

#  for((subcategory=1;subcategory<$totalSubCategories;subcategory++)); do
#    modelFile=$modelsOutput"/"$category"_"$cost"_"$expName"_"$overlap"_"$subcategory".txt"
#    time python subcatDetector.py $modelArgs $posFeatures $trainingList $featuresDir $modelFile $overlap $hardNegativeIters $labelsFile $totalSubCategories $subcategory 
#  done
#
#  ## Detections in Test Set
#  modelFile=$modelsOutput"/"$category"_"$cost"_"$expName"_"$overlap"_*.txt."$hardNegativeIters
#  resultsFile=$resultsOutput"/"$category"_"$cost"_"$expName"_"$overlap"_"$hardNegativeIters".out"
#  time python detector.py 'subcategories' "$modelFile" $testList $featuresDir $featuresExt $NMSThresholdTest $scoreThreshold $resultsFile
#  python evaluation.py 0.5 $testGroundTruth $resultsFile $resultsFile".result"
done
