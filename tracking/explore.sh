run() {
    mkdir -p $1
    CUDA_VISIBLE_DEVICES=$2 THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python recurrent_base.py --epochs=$3 --batch_size=$4 --gru_dim=$5 --learning_rate=$6 --model_name=$1/model.pkl --use_cudnn=True > $1/out.log 2> $1/err.log
}

# Add experiments to configure here
# PARAMS: outputDir device epochs batchSize GRUsize learningRate

# Minibatches
run ~/data/experiments/exp04/ 0 5 16 256 0.0005 &
run ~/data/experiments/exp05/ 1 5 64 256 0.0005 &
wait
# GRU Dim
run ~/data/experiments/exp06/ 0 5 32 128 0.0005 &
run ~/data/experiments/exp07/ 1 5 32 512 0.0005 &
wait
# Learning rate
run ~/data/experiments/exp08/ 0 5 32 256 0.001 &
run ~/data/experiments/exp09/ 1 5 32 256 0.01 &
wait

: <<'END'
# Epochs
run ~/data/experiments/exp01/ 1 32 256 0.0005
run ~/data/experiments/exp01/ 5 32 256 0.0005
run ~/data/experiments/exp01/ 10 32 256 0.0005
END
