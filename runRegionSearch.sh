# Training
modelsOutput=/home/caicedo/data/rcnn/regionSearchResults/third/models/
# Test
trainingList=/home/caicedo/data/pascal07/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt
#trainingList=/home/caicedo/data/rcnn/debug/aeroplaneDebugTrainval.txt
testList=/home/caicedo/data/pascal07/VOCdevkit/VOC2007/ImageSets/Main/test.txt
#testList=/home/caicedo/data/rcnn/debug/aeroplaneDebugTest.txt
featuresExt=fc6_neuron_cudanet_out
NMSThresholdTest=0.2
scoreThreshold="-10.0"
resultsOutput=/home/caicedo/data/rcnn/regionSearchResults/third/
# Evaluation

boxExpansion="10"
cost=0.001
expName=region_search_third
overlap=0.3
components=1
#modelArgs="C:"$cost"!"
modelType='linear' # latent | linear
#for category in aeroplane bicycle bird boat bottle bus car cat chair cow diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor; do
for category in aeroplane bird car dog ; do # 

for operation in inside ; do # big tight inside
  posFeatures="/home/caicedo/data/rcnn/regionSearchCategories3/"$operation"/"$category"."$featuresExt
  trainingBBoxes="/home/caicedo/data/rcnn/lists/2007/trainval/"$category"_gt_bboxes.txt"
  featuresDir="/home/caicedo/data/rcnn/GPUCroppedFeatures07/"
  #featuresDir="/home/caicedo/data/rcnn/FineTunedFeatures07/"
  testGroundTruth="/home/caicedo/data/pascal07/boxes/test/"$category"_test_bboxes.txt"

  modelArgs="C:"$cost"!components:"$components"!maxIters:10!featuresDir:"$featuresDir"!featuresExt:"$featuresExt"!"

  iterations=2
  ## Training
  modelFile=$modelsOutput"/"$category"_"$operation"_"$boxExpansion"_"$cost"_"$expName"_"$overlap".txt"
  #time python trainDetector.py $modelType $modelArgs $posFeatures $trainingList $featuresDir $modelFile $overlap $iterations
  #time python trainRegionDetector.py $modelType $modelArgs $posFeatures $trainingBBoxes $trainingList $featuresDir $modelFile $overlap $iterations
  NMSThresholdTest=1.0

  ## Detections in Test Set
  resultsFile=$resultsOutput"/"$category"_"$operation"_"$boxExpansion"_"$cost"_"$expName"_"$overlap"_"$iterations".out"
  #time python detector.py $modelType $modelFile"."$iterations $testList $featuresDir $featuresExt $NMSThresholdTest $scoreThreshold $resultsFile
  #python evaluation.py 0.5 $testGroundTruth $resultsFile $resultsFile".result"

  #python collectMultipleScores.py /home/caicedo/data/rcnn/regionSearchResults/third/$c"_big_10_0.001_region_search_third_0.3_2.out" /home/caicedo/data/rcnn/regionSearchResults/third/$c"_tight_10_0.001_region_search_third_0.3_2.out" /home/caicedo/data/rcnn/regionSearchResults/third/$c"_inside_10_0.001_region_search_third_0.3_2.out" /home/caicedo/data/rcnn/lists/2007/test/$c"_test_bboxes.txt" -0.5 /home/caicedo/data/rcnn/regionSearchResults/third/$c"_ranks/"o

  resultsFile=$resultsOutput"/"$category"_ranks/"$operation".txt"
  python evaluation.py OV0.9 $testGroundTruth $resultsFile $resultsFile".result"

done

done
