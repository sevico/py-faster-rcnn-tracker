 python tools/train.py \
        --gpu 0 \
        --solver model/VGG_M_1024/solver.prototxt \
        --weights /home/sigcv/Workspace/model/VGG_CNN_M_1024.caffemodel \
        --dataset ILSVRC2015 \
        --max_iters 800000
