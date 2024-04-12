gpus=0
export CUDA_VISBLE_DEVICES=$gpus
EXP_DIR=exp/tcn
WAVS=~/data/DIHARD3/third_dihard_challenge_eval/data/flac
CKPT=checkpoints/epoch=77-step=71214.ckpt
OUT=$EXP_DIR/preds/${CKPT}

python local/infer.py  --exp_dir $EXP_DIR --checkpoint_name $CKPT --wav_dir $WAVS --out_dir $OUT --gpus $gpus 