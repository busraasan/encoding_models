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
parser.add_argument("--nlp_model", default='GPT-2', choices=model_options)                
parser.add_argument("--sequence_length", type=int, default=20, help='length of context to provide to NLP model (default: 1)')
parser.add_argument("--reps_dir", required=True, default="/BRAIN/neuromod-data/static00/training_saves/", help='directory to load extracted representations from')
parser.add_argument("--output_dir", type=str, default='/BRAIN/neuromod-data/static00/training_saves/')
parser.add_argument("--recording_path", required=True, default="/BRAIN/neuromod-data/static00/narratives.stimuli/audio_files_filtered/", help='directory to load audio recordings from')
parser.add_argument("--timestamps_path", required=True, help='directory to load extracted representations from')
parser.add_argument("--use_layer", type=int, default=8)
parser.add_argument("--subject", type=int, default=2)
parser.add_argument("--n_folds", type=int, default=9)
parser.add_argument("--region", type=str, default="LANG_REGION")
parser.add_argument("--session", type=int, default=-1)


args = parser.parse_args()

model_list = {
    "BERT":"/BRAIN/neuromod-data/static00/apps/hf_cache/bert-uncased",
    "Llama-7B":"/BRAIN/neuromod-data/static00/apps/hf_cache/llama-2-7b",
    "GPT-2":"/BRAIN/neuromod-data/static00/apps/hf_cache/gpt2",
}

story_dict = {
    "Tunnelpart2Story": [8,10,12],
    "Tunnelpart1Story": [8,10,12],
    "PiemanStory": [7,9,11],
    "LucyStory": [1,3,5],
    "PrettymouthStory": [8,10,12],
    "NotthefallintactStory": [2,4,6],
    "SlumlordStory": [7,9,11],
    "BlackStory": [2,4,6],
    "ForgotStory": [1,3,5]
}

# Create the output directory
output_dir = os.path.join(
    args.output_dir,
    "all_stories",
    args.nlp_model,
    "training_results"
)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

SKIP_WORDS_START = 20
SKIP_WORDS_END = 15
TRIM_START = 0
TRIM_END = 1
experiment_TR_offset_start = 3
experiment_TR_offset_end = 5

wav_files = [
    f for f in os.listdir(args.recording_path)
    if f.endswith('.wav') and f != '21styear_audio.wav' and f != 'slumlordreach_audio.wav' and f != 'tunnel_audio.wav'
]

all_stories_dataset = []

for i, wav_file in enumerate(wav_files):

    tr_times = read_tr_times(os.path.join(args.recording_path, wav_file)) # pandas for timestamps (num_TR, start, end)
    audio_name = wav_file.split('.')[0]

    if "tunnel_part1" in audio_name.lower():
        audio_name = "tunnelpart1_audio"
    elif "tunnel_part2" in audio_name.lower():
        audio_name = "tunnelpart2_audio"
    
    word_times = read_word_times(os.path.join(args.timestamps_path, audio_name+"_timestamps.txt")) # pandas for timestamps (word, start, end)
    word_times = word_times[SKIP_WORDS_START:-SKIP_WORDS_END]
    
    story_name = audio_name.split('_')[0]
    task_name = next(
        key for key in story_dict
        if story_name.lower() in key.lower()
    )

    session = story_dict[task_name][args.session]

    # load previously saved layer representations for a model
    reps_fname = os.path.join(
                        args.reps_dir,
                        task_name,
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
                                    session='ses-' + str(session).zfill(3),
                                    task=task_name,
                                    resample_masks=False,
                                    start_trim=TRIM_START,
                                    end_trim=TRIM_END)

        all_stories_dataset.append({
            "task_name": task_name,
            "session": session,
            "tr_times": tr_times,
            "word_times": word_times,
            "audio_name": audio_name,
            "fmri_data": fmri_data,
            "layer_reps_dict": layer_reps_dict,
            "TR_aligned_features": aligned_layer_reps_dict,
        })
        
    else:
        fmri_data = load_fmri_data_fsavg_surf(datadir="/BRAIN/neuromod-data/static00/narratives.fmriprep",
                                    subject=args.subject,
                                    session=session,
                                    task=task_name,
                                    start_trim=TRIM_START,
                                    end_trim=TRIM_END)
        

def run_class_time_CV_fmri_crossval_ridge(all_stories_dataset,
                                          method = 'kernel_ridge', 
                                          lambdas = np.array([0.1,1,10,100,1000]),
                                          n_folds = n_folds,
                                          skip=skip,):
    print("Running cross validation...")

    # Just initializing array beforehand
    # Not efficient.......
    valid_voxels = ~np.any(np.isnan(all_stories_dataset[0]["fmri_data"]), axis=0)
    n_good = valid_voxels.sum()
    corrs_reduced = np.zeros((n_folds, n_good))

    for fold in range(n_folds):
        
        fmri_data, TR_aligned_features_train, fmri_data_test, TR_aligned_features_test = make_train_and_test_CV_stories(all_stories_dataset, fold, 'layer_10', experiment_TR_offset_start=experiment_TR_offset_start, experiment_TR_offset_end=experiment_TR_offset_end, TRIM_START=TRIM_START, TRIM_END=TRIM_END)

        n_samples = fmri_data.shape[0]
        n_voxels = fmri_data.shape[1] * fmri_data.shape[2] * fmri_data.shape[3]
        fmri_shape = fmri_data.shape

        # 2) build a mask of "good" voxels (those that never go NaN)
        # nonzero voxels are determined by the language region masks in native space
        valid_voxels = ~np.any(np.isnan(fmri_data), axis=0)
        n_good = valid_voxels.sum()
        print(f"Keeping {n_good}/{n_voxels} voxels (no NaNs)")

        # 3) restrict data to good voxels only
        fmri_good = fmri_data[:, valid_voxels]
        fmri_data_test_good = fmri_data_test[:, valid_voxels]

        # initialize arrays to store results
        acc = np.zeros((n_folds, n_good))
        acc_std = np.zeros((n_folds, n_good))

        all_test_data = []
        all_preds = []

        representations = np.vstack(
            [TR_aligned_features_train, TR_aligned_features_test]
        )

        n_train = TR_aligned_features_train.shape[0]
        n_test  = TR_aligned_features_test.shape[0] 

        train_indices = np.concatenate([
            np.ones(n_train, dtype=bool),
            np.zeros(n_test,  dtype=bool),
        ])
           
        _,_, pca_train_features, pca_test_features = get_nlp_features_fixed_length(seq_len, reps_fname, args.nlp_model, train_indices, representations)
        # If n_components and TR_lag = 8, then you would have (T, 8*n_components) and T is determined by the cross validation
        train_features, test_features = prepare_story_fmri_features(pca_train_features, pca_test_features, num_delay_TR)
        
        # We have our train and test features now, we will prepare the fmri data and fit the encoding model.
        train_data = fmri_good
        test_data = fmri_data_test_good

        # print("fmri data, fmri good")
        # print(fmri_data.shape, fmri_good.shape)
        # print("pca")
        # print(pca_train_features.shape, pca_test_features.shape)
        # print("train data, train features")
        # print(train_data.shape, train_features.shape)
        # print("test data, test features")
        # print(test_data.shape, test_features.shape)

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

        # train_features = np.nan_to_num(zscore(train_features))
        # test_features = np.nan_to_num(zscore(test_features))
        
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
corrs_t, _, _, preds_t, test_t = run_class_time_CV_fmri_crossval_ridge(all_stories_dataset)
fname = '/predict_sub-0{}_with_{}_layer_{}_len_{}_folds_{}_{}_session_{}'.format(args.subject, args.nlp_model, args.use_layer, args.sequence_length, n_folds, args.region, args.session)
np.save(f"{output_dir}{fname}.npy", {'corrs_t':corrs_t,'preds_t':preds_t,'test_t':test_t})
print('Saved training results → {}'.format(output_dir + fname))
