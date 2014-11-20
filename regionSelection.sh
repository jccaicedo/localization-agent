category=$1

imagesDir="/home/caicedo/data/pascalImgs/"
imgList="/home/caicedo/data/cnnPatches/lists/2007/positives/"$category"_trainval.txt"
featuresDir="/home/caicedo/data/rcnnPascal/FineTunedFeatures07/"
groundTruthFile="/home/caicedo/data/cnnPatches/lists/2007/trainval/"$category"_gt_bboxes.txt"
outputDir="/home/caicedo/data/relationsRCNN/python/features/"
featureExt="fc7"

for operation in big inside; do
  mkdir -p $outputDir/$operation
  echo $category" "$operation
  python regionSelection.py $imgList $featuresDir $groundTruthFile $outputDir"/"$operation"/" $featureExt $category $operation
done

operation=tight
mkdir -p $outputDir/$operation
python extractCNNFeaturesOnRegions2.py $groundTruthFile $imagesDir $outputDir"/"$operation"/"$category
