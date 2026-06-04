#!/bin/bash
BASE=palm
ASPL_MODE=${1:-1}
LR=0.01
EPOCH=100
SHOT=16

if [ "$ASPL_MODE" != "1" ] && [ "$ASPL_MODE" != "2" ]; then
    echo "Usage: bash scripts/run_all_datasets_palm.sh [1|2]"
    echo "  1: ASPL"
    echo "  2: ASPL+"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "Running $BASE with ASPL_MODE=$ASPL_MODE, LR=$LR, EPOCH=$EPOCH, SHOT=$SHOT"

bash scripts/beijing_opera.sh $BASE $ASPL_MODE $LR $EPOCH $SHOT
bash scripts/crema_d.sh $BASE $ASPL_MODE $LR $EPOCH $SHOT
bash scripts/esc50_actions.sh $BASE $ASPL_MODE $LR $EPOCH $SHOT
bash scripts/esc50.sh $BASE $ASPL_MODE $LR $EPOCH $SHOT
bash scripts/gt_music_genre.sh $BASE $ASPL_MODE $LR $EPOCH $SHOT
bash scripts/ns_instruments.sh $BASE $ASPL_MODE $LR $EPOCH $SHOT
bash scripts/ravdess.sh $BASE $ASPL_MODE $LR $EPOCH $SHOT
bash scripts/sesa.sh $BASE $ASPL_MODE $LR $EPOCH $SHOT
bash scripts/tut.sh $BASE $ASPL_MODE $LR $EPOCH $SHOT
bash scripts/vocal_sound.sh $BASE $ASPL_MODE $LR $EPOCH $SHOT
bash scripts/urban_sound.sh $BASE $ASPL_MODE $LR $EPOCH $SHOT
