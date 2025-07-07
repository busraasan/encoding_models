import numpy as np
from process_tr import build_delay_fir_matrix

def make_contiguous_kfold_CV_indices(
    n_samples: int,
    n_folds: int
) -> np.ndarray:
    """
    Create a simple K-fold index array by splitting samples into
    contiguous blocks (no shuffling).

    Returns
    -------
    fold_indices : np.ndarray of shape (n_samples,)

    """
    # Compute (minimum) block size per fold
    block_size = n_samples // n_folds

    # Initialize all to the last fold ID
    fold_indices = np.full(n_samples, n_folds - 1, dtype=int)

    # Assign folds 0 .. n_folds-2 to contiguous blocks of block_size
    for fold_id in range(n_folds - 1):
        start = fold_id * block_size
        end   = start + block_size
        fold_indices[start:end] = fold_id

    return fold_indices

import numpy as np
from scipy.stats import zscore

def prepare_fmri_features(
    train_features: np.ndarray,        # shape (N_train_words, D)
    test_features:  np.ndarray,        # shape (N_test_words,  D)
    word_train_indicator: np.ndarray,  # boolean mask, len = N_words
    TR_train_indicator:   np.ndarray,  # boolean mask, len = N_TRs
    tr_times,
    word_times,
    trim_TR_start:    int = 10,
    trim_TR_end:      int = 10,
    num_delay_TR:     int = 8
):
    """Return (X_TR_train, X_TR_test) design matrices for a single-run dataset."""

    # timing information 
    N_TRs      = tr_times["start"].shape[0]
    word_times = ((word_times["start"]+word_times["end"])/2).to_numpy()

    # map each word to its TR index
    words_id = np.zeros(len(word_times), dtype=int)
    for i in range(len(word_times)):
        # find the last TR time that is ≤ this word’s onset by picking the last of all TR's before word's onset.        
        words_id[i] = np.where(word_times[i] > tr_times['start'])[0][-1]

    # stitch train / test word-level features 
    all_features = np.zeros((len(word_times), train_features.shape[1]))
    all_features[word_train_indicator]  = train_features
    all_features[~word_train_indicator] = test_features

    # average word features inside each TR
    D   = all_features.shape[1] # num_hidden_dim
    X0  = np.zeros((N_TRs, D))
    for tr in range(N_TRs):
        in_this_TR = (words_id == tr)
        if in_this_TR.any():
            X0[tr] = all_features[in_this_TR].mean(axis=0)

    # add FIR lags
    X  = build_delay_fir_matrix(feature_matrix=X0, lags=np.arange(1, num_delay_TR + 1))
    # output shape: (N_TRs, D * num_delay_TR)

    # trim edges of the single run & z-score
    X = X[trim_TR_start : -trim_TR_end]          # remove first 20 & last 15 volumes
    TR_train_indicator = TR_train_indicator[trim_TR_start : -trim_TR_end]
    
    X = zscore(X, axis=0)                        # zero-mean / unit-var per regressor
    X = np.nan_to_num(X)                         # guard against all-zero cols

    # split back into train / test TRs
    return X[TR_train_indicator], X[~TR_train_indicator]
