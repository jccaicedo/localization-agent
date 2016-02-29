#!/bin/bash 
## SCRIPT TO EXECUTE TRAINING EXPERIMENTS WITH THE CONTROLLER
##
run() {
    CODE_DIR="/home/jccaicedo/localization-agent/tracking"
    mkdir -p $1
    cd $CODE_DIR

    CUDA_VISIBLE_DEVICES=$2 \
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 \
    python Controller.py \
         --epochs=$3 --batchSize=$4 --generationBatchSize=32 --gpuBatchSize=4 \
         --gruStateDim=$5 --learningRate=$6 \
         --trackerModelPath=$1/model.pkl \
         --summaryPath=/home/jccaicedo/data/simulations/CocoSummaries/cocoSummaryCategAndSideGt100Smpls10000.pkl \
         --trajectoryModelPath=$CODE_DIR/../notebooks/gmmDenseAbsoluteNormalizedOOT.pkl \
         --convFilters=$8 \
         --norm=$7 --useAttention=$9 --seqLength=$10 \
         --useCUDNN=True > $1/out.log 2> $1/err.log


    ## Three models of CNN-RNN. Put the line after --trajectoryModelPath and before --norm
    ## 1. A single conv layer with given number of filters:
    #     --convFilters=$8 \
    ## 2. A pretrained Caffe model
    #     --pretrained --caffeRoot=/home/jccaicedo/caffe/ \
    ## 3. The pretrained VGG16 model in lasagne
    #     --pretrained=lasagne --cnnModelPath=/home/jccaicedo/data/vgg16.pkl --layerKey=pool5 \

    python parseLogs.py --log_file=$1/out.log --out_file=$1/results.png \
         --batch_size=$4 --gru_dim=$5 --learning_rate=$6

    #TODO: add a call to evaluation code on validation sets

}

# Add experiments here
# PARAMS: 1.outputDir 2.device 3.epochs 4.batchSize 5.GRUsize 6.learningRate 7.norm 8.convFilters 9.visualAttention 10.sequenceLength

run ~/data/experiments/exp21/ 0 10 32 256 0.001 l2 32 no 60
run ~/data/experiments/exp22/ 0 10 32 256 0.001 l2 32 gaussian 60
run ~/data/experiments/exp23/ 0 10 32 256 0.001 l2 32 square 60


: <<'END'

## EXAMPLE WITH TWO GPUs
run ~/data/experiments/exp02/ 0 1 16 256 0.0010 &
run ~/data/experiments/exp03/ 1 1 32 256 0.0005 &
wait

END
