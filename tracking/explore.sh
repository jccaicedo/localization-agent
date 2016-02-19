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
         --epochs=$3 --batchSize=$4 \
         --gruStateDim=$5 --learningRate=$6 \
         --trackerModelPath=$1/model.pkl \
         --summaryPath=$CODE_DIR/../notebooks/cocoTrain2014Summary.pkl \
         --trajectoryModelPath=$CODE_DIR/../notebooks/gmmDenseAbsoluteNormalizedOOT.pkl \
         --pretrained --caffeRoot=/opt/caffe/ \
         --useCUDNN=True > $1/out.log 2> $1/err.log

    python parseLogs.py --log_file=$1/out.log --out_file=$1/results.png \
         --batch_size=$4 --gru_dim=$5 --learning_rate=$6

    #TODO: add a call to evaluation code on validation sets

}

# Add experiments here
# PARAMS: outputDir device epochs batchSize GRUsize learningRate
run ~/data/experiments/debug/ 0 1 32 256 0.0005 

: <<'END'

## EXAMPLE WITH TWO GPUs
run ~/data/experiments/exp02/ 0 1 16 256 0.0010 &
run ~/data/experiments/exp03/ 1 1 32 256 0.0005 &
wait

END
