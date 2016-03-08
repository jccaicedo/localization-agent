#!/bin/bash 
## SCRIPT TO EXECUTE TRAINING EXPERIMENTS WITH THE CONTROLLER
##
run() {
    CODE_DIR="/home/jccaicedo/localization-agent/tracking"
    mkdir -p $1
    cd $CODE_DIR

    if [ $3 = "base" ] ; then
        model="--modelArch=oneConvLayers --convFilters=$9"
    elif [ $3 = "caffe" ] ; then
        model="--modelArch=caffe --caffeRoot=/home/jccaicedo/caffe/"
    elif [ $3 = "lasagne" ] ; then
        model="--modelArch=lasagne --cnnModelPath=/home/jccaicedo/data/vgg16.pkl --layerKey=pool5"
    elif [ $3 = "2convl" ] ; then
        model="--modelArch=twoConvLayers --convFilters=$9"
    elif [ $3 = "3convl" ] ; then
        model="--modelArch=threeConvLayers --convFilters=$9"
    elif [ $3 = "4convl" ] ; then
        model="--modelArch=fourConvLayers --convFilters=$9"
    elif [ $3 = "5convl" ] ; then
        model="--modelArch=fiveConvLayers --convFilters=$9"
    elif [ $3 = "6convl" ] ; then
        model="--modelArch=sixConvLayers --convFilters=$9"
    fi

    CUDA_VISIBLE_DEVICES=$2 \
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 \
    stdbuf -o0 python Controller.py \
         --epochs=$4 --batchSize=$5 --generationBatchSize=32 --gpuBatchSize=4 \
         --gruStateDim=$6 --learningRate=$7 --trackerModelPath=$1/model.pkl \
         --summaryPath=/home/jccaicedo/data/simulations/CocoSummaries/cocoSummaryCategAndSideGt100Smpls10000.pkl \
         $model \
         --norm=$8 --useAttention=${10} --seqLength=${11} \
         --useCUDNN=True > $1/out.log 2> $1/err.log

#         --trajectoryModelPath=$CODE_DIR/../notebooks/gmmDenseAbsoluteNormalizedOOT.pkl \
#         --useReplayMem \
#         --imageDir=/mnt/ramdisk/ --numProcs=8 \

    python parseLogs.py --log_file=$1/out.log --out_file=$1/results.png \
         --batch_size=$5 --gru_dim=$6 --learning_rate=$7
}

# Add experiments here
# PARAMS: 1.outputDir 2.device 3.model 4.epochs 5.batchSize 6.GRUsize 7.learningRate 8.norm 9.convFilters 10.visualAttention 11.sequenceLength

run ~/data/experiments/exp81/ 0 5convl 15 32 256 0.00001 smooth_l1 4 square 2 &

: <<'END'

## EXAMPLE WITH TWO GPUs
run ~/data/experiments/exp44/ 0 3convl 10 32 256 0.00001 smooth_l1 4 square 30 &
run ~/data/experiments/exp45/ 1 3convl 10 32 256 0.001 l2 4 square 30 &
wait

END
