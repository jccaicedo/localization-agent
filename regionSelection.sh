category=$1

imgList="/home/caicedo/data/rcnn/lists/2007/positives/"$category"_trainval.txt"
featuresDir="/home/caicedo/data/rcnn/GPUCroppedFeatures07/"
groundTruthFile="/home/caicedo/data/rcnn/lists/2007/trainval/"$category"_gt_bboxes.txt"
outputDir="/home/caicedo/data/rcnn/regionSearchCategories3"
featureExt="fc6_neuron_cudanet_out"

for operation in inside; do
  mkdir -p $outputDir/$operation
  echo $category" "$operation
  python regionSelection.py $imgList $featuresDir $groundTruthFile $outputDir"/"$operation"/" $featureExt $category $operation
done
