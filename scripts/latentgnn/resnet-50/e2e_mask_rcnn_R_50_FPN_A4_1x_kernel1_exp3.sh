# >/dev/null 2>log &

# nohup sh scripts/faun_e2e_mask_rcnn/lowrank/A234_1x_layer1_branch1_exp1.sh traintwo 10.15.37.7 0 >/dev/null 2>log &
# nohup sh scripts/faun_e2e_mask_rcnn/lowrank/A234_1x_layer1_branch1_exp1.sh traintwo 10.15.37.7 1 >/dev/null 2>log &

# nohup bash scripts/faun_e2e_mask_rcnn/lowrank/A234_1x_layer1_branch1_exp1.sh train  >/dev/null 2>log &
# nohup bash scripts/faun_e2e_mask_rcnn/lowrank/A234_1x_layer1_branch1_exp1.sh test > ./save/faun_e2e_mask_rcnn/lowrank/A234_1x_layer1_branch1_exp1/test_info.log 2>&1 &

export NGPUS=4
CONFIG="configs/latentgnn/R-50/e2e_mask_rcnn_R_50_FPN_A4_1x.yaml"
DEBUG_IMAGE_BATCH=2
MASTER_ADDR=$2 # MASTER_ADDR="12.12.12.124"
RANK=$3 # RANK=0
MORE_ARGS='MODEL.FEATURE_AUG.LATENTGNN.CHANNEL_STRIDE [4,4,4,4]' 
        #   'MODEL.FEATURE_AUG.FAUN.STRIDE 8
        #    MODEL.FEATURE_AUG.FAUN.NUM_KERNEL 1
        #    MODEL.FEATURE_AUG.FAUN.NUM_LAYER 1
        #    MODEL.FEATURE_AUG.FAUN.LATENT_STRIDE 2
        #    MODEL.FEATURE_AUG.FAUN.METHOD lowrank
        #    MODEL.FEATURE_AUG.FAUN.SPATIAL_SAMPLE False
        #    MODEL.FEATURE_AUG.FAUN.QUERY_NORMALIZATION True
        #    MODEL.FEATURE_AUG.FAUN.KEY_NORMALIZATION True
        #    MODEL.FEATURE_AUG.FAUN.BN FrozenBatchNorm2d'

OUTPUT_FOLDER=./save/latentgnn/resnet-50/e2e_mask_rcnn_R_50_FPN_A4_1x_kernel1_exp3

# cd /public/sist/home/zhouk/syzhang/projects/gacn_project/maskrcnn-benchmark

train_with_onenode(){
    python -m torch.distributed.launch --nproc_per_node=$NGPUS \
        tools/train_net.py \
        --config-file $CONFIG \
        OUTPUT_DIR $OUTPUT_FOLDER $MORE_ARGS
}

train_with_twonode(){
    echo 'Rank: '$RANK 
    echo 'Maskter Addr: '$MASTER_ADDR    
    python -m torch.distributed.launch --nproc_per_node=$NGPUS \
        --nnodes=2 --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=2333 \
        tools/train_net.py \
        --config-file $CONFIG \
        OUTPUT_DIR $OUTPUT_FOLDER $MORE_ARGS
}

debug_with_onegpu(){
    python -m torch.distributed.launch --nproc_per_node=1 \
        tools/train_net.py \
        --config-file $CONFIG \
        OUTPUT_DIR $OUTPUT_FOLDER \
        SOLVER.IMS_PER_BATCH $DEBUG_IMAGE_BATCH $MORE_ARGS
}

test_with_onenode(){
    python -m torch.distributed.launch --nproc_per_node=$NGPUS \
        tools/test_net.py \
        --config-file $CONFIG \
        OUTPUT_DIR $OUTPUT_FOLDER $MORE_ARGS
}

test_with_onegpu(){
    python -m torch.distributed.launch --nproc_per_node=1 \
        tools/test_net.py \
        --config-file $CONFIG \
        OUTPUT_DIR $OUTPUT_FOLDER \
        TEST.IMS_PER_BATCH 2 $MORE_ARGS
}


if [ $1 == "train" ]; then
    echo "Start to Train the Model"
    # cd /public/sist/home/zhouk/syzhang/projects/gacn_project/maskrcnn-benchmark
    train_with_onenode
elif [ $1 == "traintwo" ]; then
    echo "Start to Train the Model"
    train_with_twonode
elif [ $1 == "test" ]; then
    echo "Start to Test the Model"
    test_with_onenode
elif [ $1 == "testone" ]; then
    echo "Start to Test the Model with One GPU"
    test_with_onegpu
elif [ $1 == "debug" ]; then
    echo "Start to Debug the Model with One GPU"
    debug_with_onegpu
else
    echo "Command Wrong !!! (\"train\",\"traintwo\",\"test\",\"testone\", or \"debug\" is valid)"
fi
