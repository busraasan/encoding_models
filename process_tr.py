import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import zscore
import time
import csv
import os
import nibabel
from sklearn.metrics.pairwise import euclidean_distances
from scipy.ndimage.filters import gaussian_filter

from utils.ridge_tools import cross_val_ridge, corr
import time as tm

def read_time_words(filepath):
    '''
    Read a timing txt file where each line is formatted as:
    word_start_end (e.g. "Okay,_0.0_0.26"), 
    and return a DataFrame with columns
    ['word', 'start', 'end']
    '''

    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # split into word and times
            try:
                word, times = line.split('_', 1)
                start_str, end_str = times.split('_', 1)
            except ValueError:
                raise ValueError(f"Line not in expected format: {line!r}")
            
            records.append({
                'word': word,
                'start': float(start_str),
                'end': float(end_str)
            })
    
    return pd.DataFrame.from_records(records, columns=['word', 'start', 'end'])

def get_TR_information(recording_path: str, TR_duration=2.0) -> dict:
    """
    Read a WAV file, compute its duration, then split into TRs of `TR_duration` seconds.
    Returns a DataFrame with columns:
      - tr_num:   0-based TR index
      - start:    start time (s) of that TR
      - end:      end time (s) of that TR (clipped to file duration)
    """
    # 1. get total duration
    with contextlib.closing(wave.open(recording_path,'r')) as wf:
        frames = wf.getnframes()
        rate   = wf.getframerate()
        duration = frames / float(rate)
    
    # 2. how many TRs to cover the full file?
    n_TRs = int(np.ceil(duration / TR_duration))
    
    # 3. build start/end arrays
    starts = np.arange(n_TRs) * TR_duration
    ends   = starts + TR_duration
    ends[-1] = min(ends[-1], duration)  # clip final TR to actual end
    
    # 4. assemble DataFrame
    tr_info = {
        'tr_num': np.arange(n_TRs),
        'start':  starts,
        'end':    ends
    }
    return tr_info


def TR_to_word_CV_ind(time_words_path: str, recording_path: str, TR_train_indicator, SKIP_WORDS=20, END_WORDS=5176):

    tr_info = get_TR_information(recording_path)
    time_words = read_time_words(time_words_path)
    runs = [1]
    offset = 20 # we have only one run TODO: think about this more
        
    word_train_indicator = np.zeros([len(time_words)], dtype=bool)    
    words_id = np.zeros([len(time_words)],dtype=int) # per word record which TR it belongs to

    # find what TR each word belongs to
    for i in range(len(time_words)):
        # find the last TR time that is ≤ this word’s onset by picking the last of all TR's before word's onset.        
        words_id[i] = np.where(time_words[i] > tr_info['start'])[0][-1]
        
        if words_id[i] <= len(tr_info) - 15:

            # check if we will use this TR as a train or test sample
            if TR_train_indicator[int(words_id[i])-offset-1] == 1:
                word_train_indicator[i] = True

    return word_train_indicator

def lanczosfun(cutoff, t, window=3):
    """Compute the lanczos function with some cutoff frequency [B] at some time [t].
    [t] can be a scalar or any shaped numpy array.
    If given a [window], only the lowest-order [window] lobes of the sinc function
    will be non-zero.
    """
    t = t * cutoff
    val = window * np.sin(np.pi*t) * np.sin(np.pi*t/window) / (np.pi**2 * t**2)
    val[t==0] = 1.0
    val[np.abs(t)>window] = 0.0

    return val

def lanczosinterp2D(data, oldtime, tr_time, window=3, cutoff_mult=1.0, rectify=False):
    """Interpolates the columns of [data], assuming that the i'th row of data corresponds to
    oldtime(i) which is the middle of the words. A new matrix with the same number of columns 
    and a number of rows given by the length of [tr_time] is returned.
    
    The time points in [newtime] are assumed to be evenly spaced, and their frequency will
    be used to calculate the low-pass cutoff of the interpolation filter.
    
    [window] lobes of the sinc function will be used. [window] should be an integer.
    """
    ## Find the cutoff frequency ## 
    cutoff = 1/np.mean(np.diff(tr_time)) * cutoff_mult
    print ("Doing lanczos interpolation with cutoff=%0.3f and %d lobes." % (cutoff, window))
    
    ## Build up sinc matrix ##
    sincmat = np.zeros((len(tr_time), len(oldtime)))
    for ndi in range(len(tr_time)):
        sincmat[ndi,:] = lanczosfun(cutoff, tr_time[ndi]-oldtime, window)
    
    if rectify:
        newdata = np.hstack([np.dot(sincmat, np.clip(data, -np.inf, 0)), 
                            np.dot(sincmat, np.clip(data, 0, np.inf))])
    else:
        ## Construct new signal by multiplying the sinc matrix by the data ##
        newdata = np.dot(sincmat, data)

    return newdata

def align_features(layer_reps_dict: dict, timestamps: pd.DataFrame, tr_times):

    # Pick timestamps for words as the middle of the start and end times.
    word_times = (timestamps["start"]+timestamps["end"])/2

    # Align the representations using lanczos filter from word timestamps to the TR timestamps.
    TR_aligned_features = []
    for layer, word_level_features in np.arange(layer_reps_dict):
        aligned_features = lanczosinterp2D(word_level_features, word_times, tr_times, window=4)
        TR_aligned_features.append(aligned_features)

