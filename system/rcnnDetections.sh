# Extract detection scores
# Training Set
python extractRCNNScores.py ~/workspace/rcnn/imdb/cache/imdb_relations_2007_trainval.mat ~/workspace/rcnn/cachedir/relations_2007_trainval/ /home/caicedo/data/relationsRCNN/scores/voc2007Train_rcnnFT/
python extractRCNNScores.py ~/workspace/rcnn/imdb/cache/imdb_voc_2007_trainval.mat ~/workspace/rcnn/cachedir/voc_2007_trainval/ /home/caicedo/data/relationsRCNN/scores/voc2007Train_rcnnFT/

# Evaluation (on test set)

for k in aeroplane bicycle bird boat bottle bus car cat chair cow diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor; do 
  python evaluation.py 0.5 /home/caicedo/data/cnnPatches/lists/2007/test/$k"_test_bboxes.txt" /home/caicedo/data/relationsRCNN/scores/rcnnResult/$k"_tight_det.out" /home/caicedo/data/relationsRCNN/scores/rcnnResult/$k"_tight_det.out.result" ; 
done > /home/caicedo/data/relationsRCNN/scores/rcnnResult/tight_results.txt

for k in aeroplane bicycle bird boat bottle bus car cat chair cow diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor; do 
  python evaluateRelations.py big0.7 /home/caicedo/data/cnnPatches/lists/2007/test/$k"_test_bboxes.txt" /home/caicedo/data/relationsRCNN/scores/rcnnResult/$k"_big_det.out" /home/caicedo/data/relationsRCNN/scores/rcnnResult/$k"_big_det.out.result" ; 
done > /home/caicedo/data/relationsRCNN/scores/rcnnResult/big_results.txt

for k in aeroplane bicycle bird boat bottle bus car cat chair cow diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor; do 
  python evaluateRelations.py inside0.7 /home/caicedo/data/cnnPatches/lists/2007/test/$k"_test_bboxes.txt" /home/caicedo/data/relationsRCNN/scores/rcnnResult/$k"_inside_det.out" /home/caicedo/data/relationsRCNN/scores/rcnnResult/$k"_inside_det.out.result" ; 
done > /home/caicedo/data/relationsRCNN/scores/rcnnResult/inside_results.txt
