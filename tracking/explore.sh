run() {
    mkdir -p $1
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python recurrent_base.py --epochs=$2 --batch_size=$3 --gru_dim=$4 --learning_rate=$5 --model_name=$1/model.pkl --use_cudnn=True > $1/out.log 2> $1/err.log
}

# Add experiments to configure here
run ~/data/experiments/exp01/ 1 32 256 0.0005

: <<'END'
# Epochs
run ~/data/experiments/exp01/ 1 32 256 0.0005
run ~/data/experiments/exp01/ 5 32 256 0.0005
run ~/data/experiments/exp01/ 10 32 256 0.0005
# Minibatches
run ~/data/experiments/exp01/ 5 16 256 0.0005
run ~/data/experiments/exp01/ 5 64 256 0.0005
# GRU Dim
run ~/data/experiments/exp01/ 5 32 128 0.0005
run ~/data/experiments/exp01/ 5 32 512 0.0005
# Learning rate
run ~/data/experiments/exp01/ 5 32 256 0.001
run ~/data/experiments/exp01/ 5 32 256 0.01
END
