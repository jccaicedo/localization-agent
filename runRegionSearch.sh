# Training
modelsOutput=/home/caicedo/data/rcnn/regionSearchResults/fifth/models/
# Test
featuresExt=fc6_neuron_cudanet_out
NMSThresholdTest=0.5 #0.2
scoreThreshold="-2.0" #"-10.0"
resultsOutput=/home/caicedo/data/rcnn/regionSearchResults/fifth/noNMS/
# Evaluation

cost=0.001
expName=region_search_fifth
declare -A OVERLAPS=( [big]=0.7 [tight]=0.5 [inside]=0.7 )
components=1
modelType='linear' # latent | linear
#for category in aeroplane bicycle bird boat bottle bus car cat chair cow diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor; do
for category in aeroplane; do

  #sh regionSelection.sh $category;

  trainingList="/home/caicedo/data/rcnn/lists/2007/debug/"$category"_training.txt"
  testList="/home/caicedo/data/rcnn/lists/2007/debug/"$category"_test.txt"

  trainingBBoxes="/home/caicedo/data/rcnn/lists/2007/trainval/"$category"_gt_bboxes.txt"
  featuresDir="/home/caicedo/data/rcnn/FineTunedFeatures07/"
  testGroundTruth="/home/caicedo/data/pascal07/boxes/test/"$category"_test_bboxes.txt"

  modelArgs="C:"$cost"!components:"$components"!maxIters:10!"
  iterations=2

  for operation in big inside ; do
    overlap=${OVERLAPS[$operation]}
    posFeatures="/home/caicedo/data/rcnn/regionSearchCategories6/"$operation"/"$category"."$featuresExt
    ## Training
    modelFile=$modelsOutput"/"$category"_"$operation"_"$cost"_"$expName"_"$overlap".txt"
    #time python trainRegionDetector.py $modelType $modelArgs $posFeatures $trainingBBoxes $trainingList $featuresDir $modelFile $overlap $iterations
    ## Detections in Test Set
    resultsFile=$resultsOutput"/"$category"_"$operation"_"$cost"_"$expName"_"$overlap"_"$iterations".out"
    time python detector.py $modelType $modelFile"."$iterations $testList $featuresDir $featuresExt $NMSThresholdTest $scoreThreshold $resultsFile
    #python evaluation.py $operation"0.8" $testGroundTruth $resultsFile $resultsFile".result"
  done

#for k in aeroplane bicycle bird boat bottle bus car cat chair cow diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor; do mkdir -p /home/caicedo/data/rcnn/regionSearchResults/sixth/$k"_ranks/"; python collectMultipleScores.py /home/caicedo/data/rcnn/regionSearchResults/sixth/$k"_big_0.001_region_search_sixth_0.5_2.out" /home/caicedo/data/rcnn/regionSearchResults/sixth/$k"_tight_0.001_region_search_sixth_0.5_2.out" /home/caicedo/data/rcnn/regionSearchResults/sixth/$k"_inside_0.001_region_search_sixth_0.5_2.out" /home/caicedo/data/rcnn/lists/2007/test/$k"_test_bboxes.txt" -0.5 /home/caicedo/data/rcnn/regionSearchResults/sixth/$k"_ranks/"; done

done
