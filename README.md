## Neuromod Narratives Encoding & Encoding Model Training

This repository provides utilities to extract language-model embeddings (features) from Neuromod Narratives stimuli and to train encoding models mapping these features to neural recordings.

File paths should be updated based on your FMRI data location.

---

## Table of Contents

1. [Overview](#overview)
2. [Data Preparation](#data-preparation)
3. [Feature Extraction](#feature-extraction)

   * [Command-line Arguments](#feature-extraction-arguments)
4. [Encoding Model Training](#encoding-model-training)

   * [Command-line Arguments](#training-arguments)
5. [Output](#logging--output)
---

## Overview

- **extract_lm_features.py**  
  Pulls contextual embeddings from your chosen NLP model (e.g. GPT-2) for each line in a stimuli file.

- **run_subject.py**  
  Aligns those embeddings to neural timestamps and fits per‐subject encoding models (ridge/regression).

- **utils.py**  
  Helper routines for FIR delays, fMRI extraction, and ROI masking.

## Data Preparation

1. **Stimuli transcriptions**: A plain-text file with one word per line (Number of lines should match with the number of words in the text. Sanity check code for this can be found at [busraasan/narratives_audio_processing](https://github.com/busraasan/narratives_audio_processing)
.
2. **Neural recordings**: Time-series fMRI data preprocessed and saved in nii.gz format.
3. **Timestamp alignment**: A CSV mapping each stimulus index to a recording timepoint.

## Feature Extraction

Example run for extracting language-model embeddings for all stimuli.

```bash
python scripts/extract_lm_features.py \
  --nlp_model GPT-2 \
  --sequence_length 20 \
  --stimuli_file stimuli_transcriptions/prettymouth_audio.txt \
  --output_dir static00/training_saves/ \
  --task_name PrettymouthStory
```

### Feature Extraction Arguments

| Argument            | Type   | Default                    | Description                                      |
| ------------------- | ------ | -------------------------- | ------------------------------------------------ |
| `--nlp_model`       | string | `GPT-2`                    | Pretrained model to use (choices: GPT-2, etc.)   |
| `--sequence_length` | int    | `20`                       | Context length (tokens) for embedding extraction |
| `--stimuli_file`    | path   | `prettymouth_audio.txt`    | Text file with one stimulus utterance per line   |
| `--output_dir`      | path   | `static00/training_saves/` | Directory to save extracted feature files by extract_lm_features.py        |
| `--task_name`       | string | `PrettymouthStory`         | Identifier for the current stimulus set          |

---

## Encoding Model Training

Train per-subject encoding models using extracted embeddings and neural recordings.

```bash
python scripts/run_subject.py \
  --reps_dir static00/training_saves/ \
  --recording_path audio_recordings/ \
  --timestamps_path stimuli_transcriptions/ \
  --subject 1 \
  --session 1 \
  --n_folds 5 \
  --nlp_model GPT-2 \
  --use_layer 0 \
```

### Training Arguments

| Argument            | Type   | Description                                                                                          |
| ------------------- | ------ | ---------------------------------------------------------------------------------------------------- |
| `--nlp_model`       | string | Which pretrained NLP model to use (must match feature‐extraction model, e.g. `GPT-2`).                |
| `--sequence_length` | int    | Number of context tokens to provide the NLP model when extracting embeddings (default: 20).          |
| `--reps_dir`        | path   | Directory containing the extracted LM feature files.                                                 |
| `--output_dir`      | path   | Directory to save encoding‐model outputs (weights, metrics, etc.).                                   |
| `--recording_path`  | path   | Directory or filepath where the audio‐stimuli recordings are stored.                                 |
| `--timestamps_path` | path   | CSV file mapping each word (or stimulus index) to its recording timepoint.                           |
| `--use_layer`       | int    | Which transformer layer’s activations to use for encoding (e.g., 8, 10).                             |
| `--subject`         | int    | Subject identifier (e.g., `1`, `2`, …).                                                              |
| `--session`         | int    | Session identifier (e.g., `1`, `2`, …; `-1` for session-agnostic runs).                              |
| `--region`          | string | Which brain region mask to use (e.g., `LANG_REGION`).                                                |
| `--n_folds`         | int    | Number of cross‐validation folds (e.g., 9 if holding out one story per fold).                        |

---

## Output

* **Model weights & metrics**: Saved under `static00/training_saves/<task_name>/<model_name>/` as a numpy object array which you can extract correlations and predictions from the dictionary entries `corrs_t` and `preds_t`.
