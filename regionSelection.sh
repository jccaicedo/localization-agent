category=$1

imagesDir="/projects/sciteam/jqh/data/pascalImgs/"
imgList="/u/sciteam/caicedor/cnnPatches/lists/2007/positives/"$category"_trainval.txt"
featuresDir="/u/sciteam/caicedor/scratch/relationsRCNN/voc2007feats/FineTunedFeatures07/"
groundTruthFile="/u/sciteam/caicedor/cnnPatches/lists/2007/trainval/"$category"_gt_bboxes.txt"
outputDir="/u/sciteam/caicedor/scratch/relationsRCNN/features/"
featureExt="fc7"

for operation in big inside; do
  mkdir -p $outputDir/$operation
  echo $category" "$operation
  python regionSelection.py $imgList $featuresDir $groundTruthFile $outputDir"/"$operation"/" $featureExt $category $operation
done

operation=tight
mkdir -p $outputDir/$operation
python extractCNNFeaturesOnRegions2.py $groundTruthFile $imagesDir $outputDir"/"$operation"/"$category
