#!/usr/bin/env bash
set -euo pipefail

# Directory where your Python script lives (adjust if needed)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Output directory (as in your argparse default)
OUTPUT_DIR="/BRAIN/neuromod-data/static00/training_saves/"

# Sequence length (default 20)
SEQ_LEN=20

# List of story task names
stories=(
#   "Tunnelpart2Story"
#   "Tunnelpart1Story"
#   "PiemanStory"
#   "LucyStory"
  "NotthefallintactStory"
  "SlumlordStory"
  "BlackStory"
  "ForgotStory"
)

for task in "${stories[@]}"; do
  # strip trailing "Story", lowercase, and build the stimuli filename
  if [[ "$task" == "NotthefallintactStory" ]]; then
    stim_file="../stimuli_transcriptions/notthefall_audio.txt"
  else
    base="${task%Story}"
    stim_file="../stimuli_transcriptions/$(echo "$base" | tr '[:upper:]' '[:lower:]')_audio.txt"
  fi

  echo "=== Running $task ==="
  python "$SCRIPT_DIR/extract_lm_features.py" \
    --nlp_model Llama-7B \
    --sequence_length "$SEQ_LEN" \
    --output_dir "$OUTPUT_DIR" \
    --stimuli_file "$stim_file" \
    --task_name "$task"
done
