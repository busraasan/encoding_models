#!/usr/bin/env bash
# run_all.sh

# fixed parameters
REPS_DIR="../training_saves"
RECORDING_PATH="/BRAIN/neuromod-data/static00/narratives.stimuli/audio_files_filtered/black_audio.wav"
TIMESTAMPS_PATH="../stimuli_transcriptions/black_audio_timestamps.txt"
TASK_NAME="BlackStory"
SUBJECT=2
NLP_MODEL="GPT-2"
LAYER_NUM="8"

for SESSION in 2 4 6; do
  for NFOLDS in 2 3 5; do
    echo "Running session=${SESSION}, n_folds=${NFOLDS}..."
    python run.py \
      --reps_dir "${REPS_DIR}" \
      --recording_path "${RECORDING_PATH}" \
      --timestamps_path "${TIMESTAMPS_PATH}" \
      --task_name "${TASK_NAME}" \
      --session "${SESSION}" \
      --subject "${SUBJECT}" \
      --n_folds "${NFOLDS}" \
      --nlp_model "${NLP_MODEL}" \
      --use_layer "${LAYER_NUM}"
  done
done
