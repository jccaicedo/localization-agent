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

    if [ ${12} = "y" ] ; then
        flow="--computeFlow"
    elif [ ${12} = "n" ] ; then
        flow=""
    fi

    CUDA_VISIBLE_DEVICES=$2 \
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 \
    stdbuf -o0 python Controller.py \
         --epochs=$4 --batchSize=$5 --generationBatchSize=32 --gpuBatchSize=4 \
         --gruStateDim=$6 --learningRate=$7 --trackerModelPath=$1/model.pkl \
         --trajectoryModelSpec random sine stretched gmm \
         --cameraTrajectoryModelSpec=still \
         --scenePathTemplate=images/scenes \
         --gmmPath=$CODE_DIR/../notebooks/gmmDenseAbsoluteNormalizedOOT.pkl \
         $flow \
         --summaryPath=/home/jccaicedo/data/simulations/CocoSummaries/cocoTrainSummaryCategAndSideGt100SmplsAllCorrected.pkl \
         $model \
         --norm=$8 --useAttention=${10} --seqLength=${11} \
         --sequenceCount=96 \
#         --useCUDNN=True > $1/out.log 2> $1/err.log

#         --imageDir=/mnt/ramdisk/ --numProcs=16 \

#    python parseLogs.py --log_file=$1/out.log --out_file=$1/results.png \
#         --batch_size=$5 --gru_dim=$6 --learning_rate=$7
}

# Add experiments here
# PARAMS: 1.outputDir 2.device 3.model 4.epochs 5.batchSize 6.GRUsize 7.learningRate 8.norm 9.convFilters 10.visualAttention 11.sequenceLength 12.flow

run ~/data/experiments/debug/ 0 6Xconvl 3 32 256 0.00005 smooth_l1 20 squareChannel 8 n 
#run ~/data/experiments/expD13/ 0 6Xconvl 1 32 256 0.00005 smooth_l1 20 squareChannel 16 n  


: <<'END'

Lenght = 4 , 8
simulator=random. [2]
Convnet {size of final feature map: 128,256,512} [3]
GRU {1,2} , {256,512} [4]
Mask {multiplicative mask, channel mask} [2]



## EXAMPLE WITH TWO GPUs
run ~/data/experiments/exp44/ 0 3convl 10 32 256 0.00001 smooth_l1 4 square 30 &
run ~/data/experiments/exp45/ 1 3convl 10 32 256 0.001 l2 4 square 30 &
wait

END
