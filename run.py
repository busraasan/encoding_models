from process_tr import *
from utils.utils import *
from extract_lm_features import *
import argparse
import os
from utils.ridge_tools import cross_val_ridge

model_options = ['BERT', 'Llama-7B']

parser = argparse.ArgumentParser()
parser.add_argument("--nlp_model", default='BERT', choices=model_options)                
parser.add_argument("--sequence_length", type=int, default=20, help='length of context to provide to NLP model (default: 1)')
parser.add_argument("--reps_dir", required=True, help='directory to load extracted representations from')
parser.add_argument("--output_dir", type=str, default='../training_saves')
parser.add_argument("--recording_path", required=True, help='directory to load extracted representations from')
parser.add_argument("--timestamps_path", required=True, help='directory to load extracted representations from')
parser.add_argument("--use_layer", type=int, default=10)
parser.add_argument("--task_name", type=str, default="BlackStory")
parser.add_argument("--subject", type=int, default=2)
parser.add_argument("--session", type=int, default=2)

args = parser.parse_args()

model_list = {
    "BERT":"google-bert/bert-base-uncased",
    "Llama-7B":"meta-llama/Llama-2-7b"
}

story_dict = {
    "PiemanStory": [7,9,11],
    "LucyStory": [1,3,5],
    "PrettymouthStory": [8,10,12],
    "NotthefallintactStory": [2,4,6],
    "Tunnelpart2Story": [8,10,12],
    "Tunnelpart1Story": [8,10,12],
    "SlumlordStory": [7,9,11],
    "BlackStory": [2,4,6],
    "ForgotStory": [1,3,5]
}

if args.task_name not in story_dict:
    raise ValueError(
        f"Unknown task_name '{args.task_name}'. "
        f"Choices: {', '.join(sorted(story_dict))}"
    )

if args.session not in story_dict[args.task_name]:
    raise ValueError(
        f"'{args.task_name}' only accepts sessions "
        f"{story_dict[args.task_name]} (got {args.session})."
    )

# Create the output directory
output_dir = os.path.join(
    args.output_dir,
    args.task_name,
    args.nlp_model,
    "training_results"
)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

SKIP_WORDS_START = 20
SKIP_WORDS_END = 15
TRIM_START = 20
TRIM_END = 15
experiment_TR_offset_start = 3
experiment_TR_offset_end = 6
session = "ses-"+str(args.session).zfill(3)
subject = "sub-0"+str(args.subject)

tr_times = read_tr_times(args.recording_path) # pandas for timestamps (num_TR, start, end)
word_times = read_word_times(args.timestamps_path) # pandas for timestamps (word, start, end)
word_times = word_times[SKIP_WORDS_START:-SKIP_WORDS_END]

# load previously saved layer representations for a model
reps_fname = os.path.join(
                    args.reps_dir,
                    args.task_name,
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
layer = args.use_layer
seq_len = args.sequence_length
layer_name = "layer_"+str(args.use_layer)
skip = 5 # skip TRs between train and test data

# TODO: fmri_data processing for encoding models
fmri_data = load_fmri_data(datadir="/BRAIN/neuromod-data/static00/narratives.fmriprep",
                            subject=subject,
                            session=session,
                            task=args.task_name,
                            start_trim=TRIM_START,
                            end_trim=TRIM_END)

def run_class_time_CV_fmri_crossval_ridge(fmri_data,
                                          tr_times = tr_times,
                                          word_times = word_times,
                                          method = 'kernel_ridge', 
                                          lambdas = np.array([0.1,1,10,100,1000]),
                                          n_folds = n_folds,
                                          skip=skip):
    print("Running cross validation...")

    n_samples = tr_times.shape[0]
    n_voxels = fmri_data.shape[1]

    fold_indices = make_contiguous_kfold_CV_indices(n_samples-(TRIM_START + TRIM_END), n_folds)

    corrs = np.zeros((n_folds, n_voxels))
    acc = np.zeros((n_folds, n_voxels))
    acc_std = np.zeros((n_folds, n_voxels))

    all_test_data = []
    all_preds = []

    fmri_data = fmri_data[experiment_TR_offset_start:-experiment_TR_offset_end]

    for fold in range(n_folds):

        # First, prepare language model representations
        train_indices = (fold_indices != fold)
        test_indices  = (fold_indices == fold)

        word_CV_indices = TR_to_word_CV_ind(word_times, tr_times, train_indices)

        representations = layer_reps_dict[layer_name]
        _,_, pca_train_features, pca_test_features = get_nlp_features_fixed_length(seq_len, reps_fname, args.nlp_model, word_CV_indices, representations)
        # If n_components and TR_lag = 8, then you would have (T, 8*n_components) and T is determined by the cross validation
        train_features, test_features = prepare_fmri_features(pca_train_features, pca_test_features, word_CV_indices, train_indices, tr_times, word_times, TRIM_START, TRIM_END)
        
        # We have our train and test features now, we will prepare the fmri data and fit the encoding model.
        train_data = fmri_data[train_indices]
        test_data = fmri_data[test_indices]

        # skip TRs between train and test data
        if fold == 0: # just remove from front end
            train_data = train_data[skip:,:]
            train_features = train_features[skip:,:]
        elif fold == n_folds-1: # just remove from back end
            train_data = train_data[:-skip,:]
            train_features = train_features[:-skip,:]
        else:
            test_data = test_data[skip:-skip,:]
            test_features = test_features[skip:-skip,:]

        # normalize representation data 
        # fmri_data already normalized
        all_test_data.append(test_data)

        train_features = np.nan_to_num(zscore(train_features))
        test_features = np.nan_to_num(zscore(test_features)) 
        
        # Train ridge regressor
        start_time = tm.time()
        weights, chosen_lambdas = cross_val_ridge(train_features, train_data, n_splits = 10, lambdas = np.array([10**i for i in range(-6,10)]), method = 'plain',do_plot = False)

        preds = np.dot(test_features, weights)
        corrs[fold, :] = corr(preds, test_data)
        all_preds.append(preds)
            
        print('fold {} completed, took {} seconds'.format(fold, tm.time()-start_time))
        del weights
        
    return corrs, acc, acc_std, np.vstack(all_preds), np.vstack(all_test_data)

# Run the K_fold nested CV for training an encoding model
corrs_t, _, _, preds_t, test_t = run_class_time_CV_fmri_crossval_ridge(fmri_data)
fname = 'predict_{}_{}_with_{}_layer_{}_len_{}'.format(subject, session, args.nlp_model, args.layer, args.sequence_length)
np.save(output_dir + fname + '.npy', {'corrs_t':corrs_t,'preds_t':preds_t,'test_t':test_t})
print(corrs_t, preds_t, test_t)
print('Saved training results → {}'.format(output_dir + fname))
