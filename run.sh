#!/usr/bin/env bash
set -euo pipefail

# ---- config you can edit ----
SCRIPT="run_subject.py"

REPS_DIR="/BRAIN/neuromod-data/static00/training_saves/"
RECORDING_PATH="/BRAIN/neuromod-data/static00/narratives.stimuli/audio_files_filtered/"
TIMESTAMPS_PATH="../stimuli_transcriptions/"
N_FOLDS=9
NLP_MODEL="Llama-7B"
USE_LAYER=8
SESSION=-1

SUBJECTS=(1 2 3 5 6)
LOG_DIR="./logs"
# -----------------------------

mkdir -p "$LOG_DIR"

for S in "${SUBJECTS[@]}"; do
  echo "=== Running subject $S ==="
  python "$SCRIPT" \
    --reps_dir "$REPS_DIR" \
    --recording_path "$RECORDING_PATH" \
    --timestamps_path "$TIMESTAMPS_PATH" \
    --subject "$S" \
    --n_folds "$N_FOLDS" \
    --nlp_model "$NLP_MODEL" \
    --use_layer "$USE_LAYER" \
    --session "$SESSION" \
    2>&1 | tee "$LOG_DIR/subject_${S}.log"
done

echo "All runs complete. Logs in: $LOG_DIR"