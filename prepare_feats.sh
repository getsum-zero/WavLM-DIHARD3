stage=0


DATA_ROOT=~/data/DIHARD3 
CONF=conf/train.yml
LABELS_DIR=~/data/DIHARD3/fa_labels

if [[ $stage -le 0 ]]; then
  echo "Getting diarization from annotation and converting to json files"
  for split in dev eval
  do
  echo $split
  python local/get_diarization_from_alignments.py $DATA_ROOT/third_dihard_challenge_$split/data/rttm data/diarization_${split}/${split}.json
  python local/get_stats_from_diarization.py data/diarization_${split}/${split}.json >> data/diarization_${split}/stats.txt
  done
fi

if [[ $stage -le 1 ]]; then
  echo "Prepping labels"
  for split in dev eval
  do
  python local/prep_labels.py  $DATA_ROOT/third_dihard_challenge_$split/data/flac data/diarization_${split}/${split}.json $CONF $LABELS_DIR/$split
  done
fi




