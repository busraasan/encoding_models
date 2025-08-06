from process_tr import *
from utils.utils import *
from extract_lm_features import *
import argparse
import os
from utils.ridge_tools import cross_val_ridge
import warnings
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore")

model_options = ['BERT', 'Llama-7B', "GPT-2"]

parser = argparse.ArgumentParser()
parser.add_argument("--nlp_model", default='BERT', choices=model_options)                
parser.add_argument("--sequence_length", type=int, default=20, help='length of context to provide to NLP model (default: 1)')
parser.add_argument("--reps_dir", required=True, help='directory to load extracted representations from')
parser.add_argument("--output_dir", type=str, default='../training_saves/')
parser.add_argument("--recording_path", required=True, help='directory to load extracted representations from')
parser.add_argument("--timestamps_path", required=True, help='directory to load extracted representations from')
parser.add_argument("--use_layer", type=int, default=10)
parser.add_argument("--task_name", type=str, default="BlackStory")
parser.add_argument("--subject", type=int, default=2)
parser.add_argument("--session", type=int, default=2)
parser.add_argument("--n_folds", type=int, default=2)
parser.add_argument("--region", type=str, default="LANG_REGION")

args = parser.parse_args()

model_list = {
    "BERT":"/BRAIN/neuromod-data/static00/apps/hf_cache/bert-uncased",
    "Llama-7B":"/BRAIN/neuromod-data/static00/apps/hf_cache/llama-2-7b",
    "GPT-2":"/BRAIN/neuromod-data/static00/apps/hf_cache/gpt2",
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

SKIP_WORDS_START = 0
SKIP_WORDS_END = 1
TRIM_START = 0
TRIM_END = 1
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
# lanczos filtering
aligned_layer_reps_dict = align_features(layer_reps_dict, word_times, tr_times)

# Define parameters for training
num_delay_TR = 8
n_folds = args.n_folds
layer = args.use_layer
seq_len = args.sequence_length
layer_name = "layer_"+str(args.use_layer)
skip = 5 # skip TRs between train and test data

if args.region == "LANG_REGION":
    fmri_data = load_fmri_data_lang_reg(datadir="/BRAIN/neuromod-data/static00/narratives.fmriprep",
                                subject=args.subject,
                                session=session,
                                task=args.task_name,
                                resample_masks=False,
                                start_trim=TRIM_START,
                                end_trim=TRIM_END)
    print("Shape of fmri_data: ", fmri_data.shape)
    print("TR times: ", tr_times.shape)
    exit()
else:
    fmri_data = load_fmri_data_fsavg_surf(datadir="/BRAIN/neuromod-data/static00/narratives.fmriprep",
                                subject=subject,
                                session=session,
                                task=args.task_name,
                                start_trim=TRIM_START,
                                end_trim=TRIM_END)
    print("Shape of fmri_data: ", fmri_data.shape)
    print("TR times: ", tr_times.shape)
    exit()


def run_class_time_CV_fmri_crossval_ridge(fmri_data,
                                          tr_times = tr_times,
                                          word_times = word_times,
                                          method = 'kernel_ridge', 
                                          lambdas = np.array([0.1,1,10,100,1000]),
                                          n_folds = n_folds,
                                          skip=skip,):
    print("Running cross validation...")

    n_samples = tr_times.shape[0]
    n_voxels = fmri_data.shape[1] * fmri_data.shape[2] * fmri_data.shape[3]
    fmri_shape = fmri_data.shape

    # 2) build a mask of "good" voxels (those that never go NaN)
    # nonzero voxels are determined by the language region masks in native space
    valid_voxels = ~np.any(np.isnan(fmri_data), axis=0)
    print("Shape of fmri_data: ", fmri_data.shape)
    print("Shape of the mask: ", valid_voxels.shape)
    n_good = valid_voxels.sum()
    print(f"Keeping {n_good}/{n_voxels} voxels (no NaNs)")

    # 3) restrict data to good voxels only
    fmri_good = fmri_data[:, valid_voxels]
    print("Shape of fmri_good: ", fmri_good.shape)

    fold_indices = make_contiguous_kfold_CV_indices(n_samples-(TRIM_START + TRIM_END), n_folds)

    # initialize arrays to store results
    corrs_reduced = np.zeros((n_folds, n_good))
    acc = np.zeros((n_folds, n_good))
    acc_std = np.zeros((n_folds, n_good))

    all_test_data = []
    all_preds = []

    fmri_good = fmri_good[experiment_TR_offset_start:-experiment_TR_offset_end]
    print("Shape of fmri_good after trimming: ")
    print(fmri_good.shape)

    for fold in range(n_folds):

        # First, prepare language model representations
        train_indices = (fold_indices != fold)
        test_indices  = (fold_indices == fold)

        word_CV_indices = TR_to_word_CV_ind(word_times, tr_times, train_indices, SKIP_WORDS_START, SKIP_WORDS_END)

        #representations = aligned_layer_reps_dict[layer_name][TRIM_START:-TRIM_END, ...]
        #print(f"Shape of representations for {layer_name}: {representations.shape}")
        # mean no
        representations = layer_reps_dict[layer_name]
        _,_, pca_train_features, pca_test_features = get_nlp_features_fixed_length(seq_len, reps_fname, args.nlp_model, word_CV_indices, representations)
        # If n_components and TR_lag = 8, then you would have (T, 8*n_components) and T is determined by the cross validation
        train_features, test_features = prepare_fmri_features(pca_train_features, pca_test_features, train_indices, word_CV_indices, tr_times, word_times, TRIM_START, TRIM_END)
        
        # We have our train and test features now, we will prepare the fmri data and fit the encoding model.
        train_data = fmri_good[train_indices]
        test_data = fmri_good[test_indices]

        # skip TRs between train and test data
        if fold == 0: # just remove from front end
            train_data = train_data[skip:]
            train_features = train_features[skip:]
        elif fold == n_folds-1: # just remove from back end
            train_data = train_data[:-skip]
            train_features = train_features[:-skip]
        else:
            test_data = test_data[skip:-skip]
            test_features = test_features[skip:-skip]


        train_features = np.nan_to_num(zscore(train_features))
        test_features = np.nan_to_num(zscore(test_features))
        
        # Train ridge regressor
        start_time = tm.time()
        weights, chosen_lambdas = cross_val_ridge(train_features, train_data)

        preds = np.dot(test_features, weights)
        corrs_reduced[fold, :] = corr(preds, test_data)
        all_preds.append(preds)
        all_test_data.append(test_data)
            
        print('fold {} completed, took {} seconds\n'.format(fold, tm.time()-start_time))
        print("Mean correlation: ")
        print(np.mean(corrs_reduced[fold, :]))
        print("\n")
        del weights

    # 4) scatter back into full‐voxel array
    corrs = np.full((n_folds, fmri_shape[1], fmri_shape[2], fmri_shape[3]), np.nan, dtype=float)
    corrs[:, valid_voxels] = corrs_reduced
        
    return corrs, acc, acc_std, np.vstack(all_preds), np.vstack(all_test_data)

# Run the K_fold nested CV for training an encoding model
corrs_t, _, _, preds_t, test_t = run_class_time_CV_fmri_crossval_ridge(fmri_data)
fname = '/predict_{}_{}_with_{}_layer_{}_len_{}_folds_{}_{}'.format(subject, session, args.nlp_model, args.use_layer, args.sequence_length, n_folds, args.region)
np.save(f"{output_dir}{fname}.npy", {'corrs_t':corrs_t,'preds_t':preds_t,'test_t':test_t})
print('Saved training results → {}'.format(output_dir + fname))
