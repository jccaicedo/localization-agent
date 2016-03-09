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
    elif [ $3 = "5Xconvl" ] ; then
        model="--modelArch=fiveXConvLayers --convFilters=$9"
    elif [ $3 = "6Xconvl" ] ; then
        model="--modelArch=sixXConvLayers --convFilters=$9"
    fi

    CUDA_VISIBLE_DEVICES=$2 \
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 \
    stdbuf -o0 python Controller.py \
         --epochs=$4 --batchSize=$5 --generationBatchSize=32 --gpuBatchSize=4 \
         --gruStateDim=$6 --learningRate=$7 --trackerModelPath=$1/model.pkl \
         --imageDir=/mnt/ramdisk/ --numProcs=16 \
         --useReplayMem \
         --summaryPath=/home/jccaicedo/data/simulations/CocoSummaries/cocoTrainSummaryCategAndSideGt100SmplsAll.pkl \
         $model \
         --norm=$8 --useAttention=${10} --seqLength=${11} \
         --useCUDNN=True > $1/out.log 2> $1/err.log

#         --trajectoryModelPath=$CODE_DIR/../notebooks/gmmDenseAbsoluteNormalizedOOT.pkl \

    python parseLogs.py --log_file=$1/out.log --out_file=$1/results.png \
         --batch_size=$5 --gru_dim=$6 --learning_rate=$7
}

runFlow() {
    CODE_DIR="/home/jccaicedo/localization-agent/tracking"
    mkdir -p $1
    cd $CODE_DIR

    if [ $3 = "base" ] ; then
        model="--modelArch=base --convFilters=$9"
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
    elif [ $3 = "5Xconvl" ] ; then
        model="--modelArch=fiveXConvLayers --convFilters=$9"
    elif [ $3 = "6Xconvl" ] ; then
        model="--modelArch=sixXConvLayers --convFilters=$9"
    fi

    CUDA_VISIBLE_DEVICES=$2 \
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 \
    stdbuf -o0 python Controller.py \
         --epochs=$4 --batchSize=$5 --generationBatchSize=32 --gpuBatchSize=4 \
         --gruStateDim=$6 --learningRate=$7 --trackerModelPath=$1/model.pkl \
         --imageDir=/mnt/ramdisk/ --numProcs=16 \
         --useReplayMem --computeFlow \
         --summaryPath=/home/jccaicedo/data/simulations/CocoSummaries/cocoTrainSummaryCategAndSideGt100SmplsAll.pkl \
         --trajectoryModelPath=$CODE_DIR/../notebooks/gmmDenseAbsoluteNormalizedOOT.pkl \
         $model \
         --norm=$8 --useAttention=${10} --seqLength=${11} \
         --useCUDNN=True > $1/out.log 2> $1/err.log

    python parseLogs.py --log_file=$1/out.log --out_file=$1/results.png \
         --batch_size=$5 --gru_dim=$6 --learning_rate=$7
}


# Add experiments here
# PARAMS: 1.outputDir 2.device 3.model 4.epochs 5.batchSize 6.GRUsize 7.learningRate 8.norm 9.convFilters 10.visualAttention 11.sequenceLength

# Exp96: Repetition of exp89 with new validation set (without bug)
# Exp97: Optical flow and multiplicative mask
# Exp98: square mask channel and optical flow
#run ~/data/experiments/exp98/ 1 6Xconvl 15 32 256 0.0001 smooth_l1 2 squareChannel 2 &
# Exp99: only mask channel
#run ~/data/experiments/exp99/ 0 6Xconvl 15 32 256 0.0001 smooth_l1 2 squareChannel 2 &

# NEW ROUND OF EXPERIMENTS

#run ~/data/experiments/exp100/ 0 6Xconvl 15 32 256 0.0001 smooth_l1 2 squareChannel 2 & # Flow
#sleep 120
#run ~/data/experiments/exp101/ 1 6Xconvl 15 32 256 0.00001 smooth_l1 2 square 2 & # No flow
#wait

# This one eat up the previous experiment :(
#runFlow ~/data/experiments/exp101/ 0 6Xconvl 15 32 256 0.00001 smooth_l1 2 squareChannel 2 & # Flow
#sleep 120
#run ~/data/experiments/exp102/ 1 6Xconvl 15 32 256 0.00001 smooth_l1 2 squareChannel 2 & #No flow!!
#wait

#runFlow ~/data/experiments/exp103/ 0 6Xconvl 20 32 256 0.0001 smooth_l1 2 squareChannel 4 & # Flow
#sleep 120
#run ~/data/experiments/exp104/ 1 6Xconvl 20 32 256 0.00001 smooth_l1 2 squareChannel 4 & # No flow!!
#wait

##
## The following experiments use the new summary AND the new trajectory models all combined
##
sleep 5500
runFlow ~/data/experiments/exp105/ 0 6Xconvl 15 32 256 0.0001 smooth_l1 2 squareChannel 2 & # Using new summary and various simulation models
sleep 120
runFlow ~/data/experiments/exp106/ 1 6Xconvl 20 32 256 0.0001 smooth_l1 2 squareChannel 4 & # Using new summary and various simulation models

: <<'END'

## EXAMPLE WITH TWO GPUs
run ~/data/experiments/exp44/ 0 3convl 10 32 256 0.00001 smooth_l1 4 square 30 &
run ~/data/experiments/exp45/ 1 3convl 10 32 256 0.001 l2 4 square 30 &
wait

END
