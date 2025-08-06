import numpy as np
from process_tr import build_delay_fir_matrix
import nibabel as nib
from scipy import signal
from scipy.ndimage import gaussian_filter
from scipy.stats import zscore
from nilearn import datasets, surface
import os
from nilearn.datasets import load_mni152_gm_mask
from nilearn.masking import apply_mask
import copy

import numpy as np

def make_train_and_test_CV_stories(all_stories_dataset, fold_id, layer, experiment_TR_offset_start=0, experiment_TR_offset_end=1, TRIM_START=20, TRIM_END=15):
    """
    Concatenate all but the last story into train X/y,
    and use the last story as test X/y.

    Parameters
    ----------
    all_stories_dataset : list of dict
        Each dict must have keys:
          - "fmri_data":       array of shape (T_i, V)
          - "TR_aligned_features": array of shape (T_i, F)

    Returns
    -------
    X_train : np.ndarray, shape (sum(T_i) for i < N, V)
    y_train : np.ndarray, shape (sum(T_i) for i < N, F)
    X_test  : np.ndarray, shape (T_N, V)
    y_test  : np.ndarray, shape (T_N, F)
    """

    # Number of stories
    N = len(all_stories_dataset)
    if N < 2:
        raise ValueError("Need at least 2 stories to split into train/test")

    # Split into all-but-last and last
    test_story = copy.deepcopy(all_stories_dataset[fold_id])

    X_test  = test_story["fmri_data"][experiment_TR_offset_start:-experiment_TR_offset_end]
    y_test  = test_story["TR_aligned_features"][layer][TRIM_START:-TRIM_END]

    if X_test.shape[0] != y_test.shape[0]:
        if X_test.shape[0] - y_test.shape[0] == 1:
            X_test = X_test[:-1]
        else:
            y_test = y_test[:-1]

    # Build train stories by skipping the test index
    raw_train = all_stories_dataset[:fold_id] + all_stories_dataset[fold_id+1:]
    train_stories = copy.deepcopy(raw_train)

    # there is something wrong with the last TR, so we remove it
    for s in train_stories:

        s["fmri_data"] = s["fmri_data"][experiment_TR_offset_start:-experiment_TR_offset_end]
        s["TR_aligned_features"][layer] = s["TR_aligned_features"][layer][TRIM_START:-TRIM_END]

        if s["fmri_data"].shape[0] != s["TR_aligned_features"][layer].shape[0]:

            #print(s["fmri_data"].shape[0], s["TR_aligned_features"][layer].shape[0])

            if s["TR_aligned_features"][layer].shape[0] - s["fmri_data"].shape[0] == 1:
                s["TR_aligned_features"][layer] = s["TR_aligned_features"][layer][:-1]
            else:
                s["fmri_data"] = s["fmri_data"][:-1]

    X_train = np.concatenate([s["fmri_data"] for s in train_stories], axis=0)
    y_train = np.concatenate([s["TR_aligned_features"][layer] for s in train_stories], axis=0)

    print(f"Train: X={X_train.shape}")
    print(f" Test: X={X_test.shape}")

    return X_train, y_train, X_test, y_test


def smooth_run_not_masked(data_4d, smoothing_factor):
    """
    Applies spatial Gaussian smoothing to a 4D fMRI time series.
    
    Parameters:
        data_4d (numpy.ndarray): fMRI data of shape (X, Y, Z, T)
        smoothing_factor (float): Gaussian kernel sigma in voxel units
    
    Returns:
        numpy.ndarray: Smoothed 4D data of same shape
    """
    smoothed_data = np.zeros_like(data_4d)
    
    for t in range(data_4d.shape[3]):
        smoothed_data[..., t] = gaussian_filter(data_4d[..., t], sigma=smoothing_factor)
    
    return smoothed_data

def load_and_process(file, start_trim = 20, end_trim = 15, do_detrend=True, smoothing_factor = 1,
                     do_zscore = True):
    dat = nib.load(file).get_fdata()
    # very important to transpose otherwise data and brain surface don't match
    dat = dat.T
    #trimming
    if end_trim>0:
        dat = dat[start_trim:-end_trim]
    else: # to avoid empty error when end_trim = 0
        dat = dat[start_trim:]
    # detrending
    if do_detrend:
        dat = signal.detrend(np.nan_to_num(dat),axis =0)
    # smoothing
    if smoothing_factor>0:
        # need to zscore before smoothing
        dat = np.nan_to_num(zscore(dat))
        dat = smooth_run_not_masked(dat, smoothing_factor)
    # zscore
    if do_zscore:
        dat = np.nan_to_num(zscore(dat))
    return dat


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

def prepare_story_fmri_features(
    train_features: np.ndarray,        # shape (N_train_words, D)
    test_features:  np.ndarray,        # shape (N_test_words,  D)
    num_delay_TR:     int = 8,
):
    X_train  = build_delay_fir_matrix(feature_matrix=train_features, lags=np.arange(1, num_delay_TR + 1))
    X_train = zscore(X_train, axis=0)                        # zero-mean / unit-var per regressor
    X_train = np.nan_to_num(X_train)

    X_test  = build_delay_fir_matrix(feature_matrix=test_features, lags=np.arange(1, num_delay_TR + 1))
    X_test = zscore(X_test, axis=0)                        # zero-mean / unit-var per regressor
    X_test = np.nan_to_num(X_test)

    return X_train, X_test


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
    X = zscore(X, axis=0)                        # zero-mean / unit-var per regressor
    X = np.nan_to_num(X)                         # guard against all-zero cols

    # split back into train / test TRs
    return X[TR_train_indicator], X[~TR_train_indicator]

def load_fmri_data(datadir, subject, session, task, start_trim=20, end_trim=15):
    fname = '{}/{}/{}/func/{}_{}_task-{}_space-MNI152NLin2009cAsym_desc-preproc_part-mag_bold.nii.gz'.format(datadir, subject, session, subject, session, task)
    bold_img = load_and_process(file=fname, start_trim=start_trim, end_trim=end_trim)
    print("FMRI dataset all TRs: ", bold_img.shape[0]+start_trim+end_trim)
    bold_2d = bold_img.reshape(bold_img.shape[0], -1)
    return bold_2d

def load_fmri_data_fsavg_surf(
    datadir: str,
    subject: str,
    session: str,
    task: str,
    start_trim: int = 20,
    end_trim:   int = 15
):
    """
    Load the preprocessed BOLD, trim first/last TRs, and project to fsaverage surface.
    Returns (surf_left, surf_right, surf_combined), each of shape (T, n_vertices).
    """
    # 1) build path & load+trim into a 4D numpy array shape (T, X, Y, Z)
    fname = os.path.join(
        datadir, subject, session, "func",
        f"{subject}_{session}_task-{task}_"
        "space-MNI152NLin2009cAsym_desc-preproc_part-mag_bold.nii.gz"
    )
    bold_4d = load_and_process(
        file=fname,
        start_trim=start_trim,
        end_trim=end_trim
    )  # → assumed shape (T, X, Y, Z)
    print("FMRI dataset all TRs: ", bold_4d.shape[0])

    # 2) grab the affine from the original NIfTI
    affine = nib.load(fname).affine

    # 3) reorder to (X, Y, Z, T) for vol_to_surf
    volume_data = np.moveaxis(bold_4d, 0, -1)

    # 4) fetch meshes from fsaverage
    fs = datasets.fetch_surf_fsaverage()
    mesh_l = surface.load_surf_mesh(fs["pial_left"])   # (coords, faces)
    mesh_r = surface.load_surf_mesh(fs["pial_right"])

    # 5) allocate output arrays
    T = volume_data.shape[3]
    n_l = mesh_l[0].shape[0]
    n_r = mesh_r[0].shape[0]
    surf_l = np.zeros((T, n_l))
    surf_r = np.zeros((T, n_r))

    # 6) project each timepoint
    for t in range(T):
        img_t = nib.Nifti1Image(volume_data[..., t], affine)
        surf_l[t] = surface.vol_to_surf(img_t, mesh_l)
        surf_r[t] = surface.vol_to_surf(img_t, mesh_r)


    # 7) combine & return
    surf_combined = np.column_stack((surf_l, surf_r))
    return surf_combined


def load_fmri_data_gm(
    datadir: str,
    subject: str,
    session: str,
    task: str,
    start_trim: int = 20,
    end_trim:   int = 15
):
    """
    Load preproc BOLD, trim first/last TRs, and mask to MNI152 gray-matter.
    Returns:
      - gm_data: array of shape (T, V_gm)
      - T: number of time-points
      - V_gm: number of gray-matter voxels
    """
    # 1) build filename and load the 4D image
    fname = os.path.join(
        datadir, subject, session, "func",
        f"{subject}_{session}_task-{task}_"
        "space-MNI152NLin2009cAsym_desc-preproc_part-mag_bold.nii.gz"
    )
    img = nib.load(fname)  # Niimg-like

    # 2) trim first/last TRs by slicing the data array
    # data4d = img.get_fdata()            # shape (X, Y, Z, T_all)
    # data4d = data4d[..., start_trim:-end_trim]  # shape (X,Y,Z,T)
    # T = data4d.shape[3]
    data4d = load_and_process(
        file=fname,
        start_trim=start_trim,
        end_trim=end_trim
    )  # → assumed shape (T, X, Y, Z)
    T = data4d.shape[0]

    # 3) rebuild a trimmed Nifti1Image so that apply_mask sees correct header
    trimmed_img = nib.Nifti1Image(data4d, img.affine, img.header)

    # 4) load the MNI152 gray‐matter mask
    gm_mask_img = load_mni152_gm_mask()  

    # 5) extract only GM voxels → array of shape (T, V_gm)
    #    apply_mask will flatten spatial dims and select voxels where mask>0
    gm_data = apply_mask(trimmed_img, gm_mask_img)

    # 6) return
    V_gm = gm_data.shape[1]
    return gm_data


def load_fmri_data_lang_reg(
    datadir: str,
    subject: str,
    session: str,
    task: str,
    resample_masks: bool = False,
    start_trim=20,
    end_trim=15
) -> np.ndarray:
    """
    Load a functional or ISC NIfTI file (3-D or 4-D) and keep only the
    voxels that fall inside a set of **language-ROI masks**.

    Returns
    -------
    lang_data : ndarray, shape (T, V_lang)
        • `T` = number of time-points (``1`` if the input is 3-D).  
        • `V_lang` = number of voxels in the union of all masks.
    """
    # ------------------------------------------------------------------
    # 1) load the functional / ISC image
    # ------------------------------------------------------------------
    subject=str(subject)
    source_folder = "/BRAIN/neuromod-data/static00/anat.atlases/tpl-sub0"+subject+"T1w/"
    mask_paths = [
        source_folder+"tpl-sub0"+subject+"T1w_res-func_atlas-langToneva_label-AngularG_mask.nii.gz",
        source_folder+"tpl-sub0"+subject+"T1w_res-func_atlas-langToneva_label-AntTemp_mask.nii.gz",
        source_folder+"tpl-sub0"+subject+"T1w_res-func_atlas-langToneva_label-dmpfc_mask.nii.gz",
        source_folder+"tpl-sub0"+subject+"T1w_res-func_atlas-langToneva_label-IFG_mask.nii.gz",
        source_folder+"tpl-sub0"+subject+"T1w_res-func_atlas-langToneva_label-IFGorb_mask.nii.gz",
        source_folder+"tpl-sub0"+subject+"T1w_res-func_atlas-langToneva_label-MFG_mask.nii.gz",
        source_folder+"tpl-sub0"+subject+"T1w_res-func_atlas-langToneva_label-pCingulate_mask.nii.gz",
        source_folder+"tpl-sub0"+subject+"T1w_res-func_atlas-langToneva_label-PostTemp_mask.nii.gz",
    ]

    func_path = f"/BRAIN/neuromod-data/static00/narratives.fmriprep/sub-0{subject}/{session}/func/sub-0{subject}_{session}_task-{task}_space-T1w_desc-preproc_part-mag_bold.nii.gz"
    
    func_data = load_and_process(
        file=func_path,
        start_trim=start_trim,
        end_trim=end_trim
    )  # → assumed shape (T, X, Y, Z)
    T = func_data.shape[0]
    func_data = func_data.T

    # Create an output array to hold the mask data, same shape as the functional data
    output_data = np.full(func_data.shape, np.nan)

    combined_mask = None
    affine = None
    header = None

    for mask_path in mask_paths:
        # Load the mask
        atlas_mask_img = nib.load(mask_path)
        atlas_mask_data = atlas_mask_img.get_fdata().astype(bool)

        # Store affine & header for saving later
        if affine is None:
            affine = atlas_mask_img.affine
            header = atlas_mask_img.header

        # Build combined mask (union)
        if combined_mask is None:
            combined_mask = atlas_mask_data.copy()
        else:
            combined_mask |= atlas_mask_data  # logical OR to get union

        output_data[atlas_mask_data] = func_data[atlas_mask_data]

    # Now save the combined mask
    # output_mask_path = "./combined_mask_sub-0{subject}.nii.gz"
    # combined_mask_img = nib.Nifti1Image(combined_mask.astype(np.uint8), affine, header)
    # nib.save(combined_mask_img, output_mask_path)
    # print(f"Combined mask saved to {output_mask_path}")

    return output_data.T
