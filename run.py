from process_tr import *
from utils.utils import *
from extract_lm_features import *
import argparse
import os

model_options = ['BERT', 'Llama-7B']

parser = argparse.ArgumentParser()
parser.add_argument("--nlp_model", default='BERT', choices=model_options)                
parser.add_argument("--sequence_length", type=int, default=20, help='length of context to provide to NLP model (default: 1)')
parser.add_argument("--reps_dir", required=True, help='directory to load extracted representations from')
parser.add_argument("--recording_path", required=True, help='directory to load extracted representations from')
parser.add_argument("--timestamps_path", required=True, help='directory to load extracted representations from')
parser.add_argument("--use_layer", type=int, default=10)

args = parser.parse_args()

model_list = {
    "BERT":"google-bert/bert-base-uncased",
    "Llama-7B":"meta-llama/Llama-2-7b"
}

SKIP_WORDS_START = 20
SKIP_WORDS_END = 15

tr_times = read_tr_times(args.recording_path) # pandas for timestamps (num_TR, start, end)
word_times = read_word_times(args.timestamps_path) # pandas for timestamps (word, start, end)
word_times = word_times[SKIP_WORDS_START:-SKIP_WORDS_END]

# load previously saved layer representations for a model
reps_fname = os.path.join(
                    args.reps_dir,
                    args.nlp_model,
                    f"{args.nlp_model}_length_{args.sequence_length}_all_layers.npz"
                )
layer_reps_dict = load_layer_representations_dict(reps_fname)

# If SKIP_WORDS_START is true and we are trimming the input, also trim the feature vectors of those words
for layer_name, feature_vector in layer_reps_dict.items():
    layer_reps_dict[layer_name] = feature_vector[SKIP_WORDS_START:-SKIP_WORDS_END, ...]

# align word representations with the TRs
TR_aligned_features = align_features(layer_reps_dict, word_times, tr_times)

# Define parameters for training
num_delay_TR = 8
n_folds = 3
n_samples = tr_times.shape[0]
layer = args.use_layer
seq_len = args.sequence_length
layer_name = "layer_"+str(args.use_layer)

fold_indices = make_contiguous_kfold_CV_indices(n_samples, n_folds)

for fold in range(n_folds):

    # First, prepare language model representations

    test_indices  = (fold_indices == fold)
    train_indices = (fold_indices != fold)

    word_CV_indices = TR_to_word_CV_ind(word_times, tr_times, train_indices)

    representations = layer_reps_dict[layer_name]
    _,_, pca_train_features, pca_test_features = get_nlp_features_fixed_length(seq_len, reps_fname, args.nlp_model, word_CV_indices, representations)
    # If n_components and TR_lag = 8, then you would have (T, 8*n_components) and T is determined by the cross validation
    train_features, test_features = prepare_fmri_features(pca_train_features, pca_test_features, word_CV_indices, train_indices, tr_times, word_times)
    
    # We have our train and test features now, we will prepare the fmri data and fit the encoding model.